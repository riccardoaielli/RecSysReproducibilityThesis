#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""


from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO
import copy
import numpy as np
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator
from normalized_adjacency_matrix import normalized_adjacency_matrix, from_sparse_to_tensor

CHUNK_SIZE_FOR_SPMM = 10000


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking
    Args:
        - gamma(float): Small value to avoid division by zero
    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.
    Examples::
        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss



class RGCF(nn.Module):

    def __init__(self,
                 interaction_matrix,
                 embedding_size,
                 n_layers,
                 reg_weight,
                 prune_threshold,
                 MIM_weight,
                 tau,
                 aug_ratio,
                 device,
                 ):
        super(RGCF, self).__init__()

        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.device = device

        # generate interaction_matrix
        # self.inter_matrix_type = config['inter_matrix_type']
        # value_field = self.RATING if self.inter_matrix_type == 'rating' else None
        # self.interaction_matrix = dataset.inter_matrix(form='coo', value_field=value_field).astype(np.float32)
        self.interaction_matrix = sps.coo_matrix(interaction_matrix)
        self.n_users, self.n_items = self.interaction_matrix.shape

        # define layers
        # self.user_linear = torch.nn.Linear(in_features=self.n_items, out_features=self.embedding_size, bias=False)
        # self.item_linear = torch.nn.Linear(in_features=self.n_users, out_features=self.embedding_size, bias=False)
        self.user_linear = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embedding_size)))
        self.item_linear = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embedding_size)))

        # define loss
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        # adj_matrix = self.get_adj_mat(self.interaction_matrix.tocoo())
        # self.norm_adj_matrix = self.get_norm_mat(adj_matrix).to(self.device)

        self.norm_adj_matrix = from_sparse_to_tensor(normalized_adjacency_matrix(interaction_matrix)).to(self.device)

        # for learn adj
        # self.spmm = spmm
        # self.special_spmm = SpecialSpmm() if self.spmm == 'spmm' else torch.sparse.mm

        self.prune_threshold = prune_threshold
        self.MIM_weight = MIM_weight
        self.tau = tau
        self.aug_ratio = aug_ratio
        self.pool_multi = 10

        self.for_learning_adj()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        # self.apply(self._init_weights)
        # self.apply(xavier_uniform_initialization)
        # self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def for_learning_adj(self):
        self.adj_indices = self.norm_adj_matrix.indices()
        self.adj_shape = self.norm_adj_matrix.shape
        self.adj = self.norm_adj_matrix

        inter_data = torch.FloatTensor(self.interaction_matrix.data).to(self.device)
        inter_user = torch.LongTensor(self.interaction_matrix.row).to(self.device)
        inter_item = torch.LongTensor(self.interaction_matrix.col).to(self.device)
        inter_mask = torch.stack([inter_user, inter_item], dim=0)

        self.inter_spTensor = torch.sparse.FloatTensor(inter_mask, inter_data, self.interaction_matrix.shape).to(self.device).coalesce()
        self.inter_spTensor_t = self.inter_spTensor.t().coalesce()

        self.inter_indices = self.inter_spTensor.indices()
        self.inter_shape = self.inter_spTensor.shape

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         normal_(module.weight.data, 0, 0.01)
    #         if module.bias is not None:
    #             module.bias.data.fill_(0.0)
    #     elif isinstance(module, nn.Embedding):
    #         normal_(module.weight.data, 0, 0.01)

    # Returns: torch.FloatTensor: The embedding tensor of all user, shape: [n_users, embedding_size]
    def get_all_user_embedding(self):
        all_user_embedding = torch.sparse.mm(self.inter_spTensor, self.user_linear)
        return all_user_embedding

    def get_all_item_embedding(self):
        all_item_embedding = torch.sparse.mm(self.inter_spTensor_t, self.item_linear)
        return all_item_embedding

    # Generate adj
    # def get_adj_mat(self, inter_M, data=None):
    #     if data is None:
    #         data = [1] * inter_M.data
    #     inter_M_t = inter_M.transpose()
    #     A = sps.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
    #     data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), data))
    #     data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), data)))
    #     A._update(data_dict)  # dok_matrix
    #     return A
    #
    # def get_norm_mat(self, A):
    #     r""" A_{hat} = D^{-0.5} \times A \times D^{-0.5} """
    #     # norm adj matrix
    #     sumArr = (A > 0).sum(axis=1)
    #     # add epsilon to avoid divide by zero Warning
    #     diag = np.array(sumArr.flatten())[0] + 1e-7
    #     diag = np.power(diag, -0.5)
    #     D = sps.diags(diag)
    #     L = D * A * D
    #     # covert norm_adj matrix to tensor
    #     SparseL = sp2tensor(L)
    #     return SparseL

    # Learn adj
    def sp_cos_sim(self, a, b, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        L = self.inter_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            batch_indices = self.inter_indices[:, idx:idx + CHUNK_SIZE]

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods

        return torch.sparse_coo_tensor(self.inter_indices, sims, size=self.interaction_matrix.shape,
                                       dtype=sims.dtype, requires_grad = False).coalesce()

    def get_sim_mat(self):
        user_feature = self.get_all_user_embedding().to(self.device)
        item_feature = self.get_all_item_embedding().to(self.device)
        sim_inter = self.sp_cos_sim(user_feature, item_feature)
        return sim_inter

    def inter2adj(self, inter):
        inter_t = inter.t().coalesce()
        data = inter.values()
        data_t = inter_t.values()
        adj_data = torch.cat([data, data_t], dim=0)
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(self.device).coalesce()
        return adj

    def get_sim_adj(self, pruning):
        sim_mat = self.get_sim_mat()
        sim_adj = self.inter2adj(sim_mat)

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)
        pruned_sim_value = torch.where(sim_value < pruning, torch.zeros_like(sim_value),
                                       sim_value) if pruning > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, self.adj_shape).to(self.device).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value, self.adj_shape).to(self.device).coalesce()

        return normal_sim_adj

    def ssl_triple_loss(self, z1: torch.Tensor, z2: torch.Tensor, all_emb: torch.Tensor):
        norm_emb1 = F.normalize(z1)
        norm_emb2 = F.normalize(z2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def cal_cos_sim(self, u_idx, i_idx, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        user_feature = self.get_all_user_embedding().to(self.device)
        item_feature = self.get_all_item_embedding().to(self.device)

        L = u_idx.shape[0]
        sims = torch.zeros(L, dtype=user_feature.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            a_batch = torch.index_select(user_feature, 0, u_idx[idx:idx + CHUNK_SIZE])
            b_batch = torch.index_select(item_feature, 0, i_idx[idx:idx + CHUNK_SIZE])
            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods
        return sims

    def get_aug_adj(self, adj):
        # random sampling
        aug_user = torch.from_numpy(np.random.choice(self.n_users,
                                                     int(adj._nnz() * self.aug_ratio * 0.5 * self.pool_multi))).to(self.device).long()
        aug_item = torch.from_numpy(np.random.choice(self.n_items,
                                                     int(adj._nnz() * self.aug_ratio * 0.5 * self.pool_multi))).to(self.device).long()

        # consider reliability
        cos_sim = self.cal_cos_sim(aug_user, aug_item)
        val, idx = torch.topk(cos_sim, int(adj._nnz() * self.aug_ratio * 0.5))
        aug_user = aug_user[idx]
        aug_item = aug_item[idx]

        aug_indices = torch.stack([aug_user, aug_item + self.n_users], dim=0)
        aug_value = torch.ones_like(aug_user) * torch.median(adj.values())
        sub_aug = torch.sparse.FloatTensor(aug_indices, aug_value, adj.shape).to(self.device).coalesce()
        aug = sub_aug + sub_aug.t()
        aug_adj = (adj + aug).coalesce()

        aug_adj_indices = aug_adj.indices()
        diags = torch.sparse.sum(aug_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -0.5)
        diag_lookup = diags[aug_adj_indices[0, :]]

        value_DA = diag_lookup.mul(aug_adj.values())
        normal_aug_value = value_DA.mul(diag_lookup)
        normal_aug_adj = torch.sparse.FloatTensor(aug_adj_indices, normal_aug_value, self.norm_adj_matrix.shape).to(self.device).coalesce()
        return normal_aug_adj

    # Train
    def forward(self, pruning=0.0, epoch_idx=0):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.adj = self.norm_adj_matrix if pruning < 0.0 else self.get_sim_adj(pruning)
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def ssl_forward(self, epoch_idx=0):
        user_embeddings = self.get_all_user_embedding()
        item_embeddings = self.get_all_item_embedding()
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.aug_adj = self.get_aug_adj(self.adj.detach())
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.aug_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, batch, epoch_idx):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # obtain embedding
        user, pos_item, neg_item = batch
        # user = interaction[self.USER_ID]
        # pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward(pruning=self.prune_threshold, epoch_idx=epoch_idx)
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)
        loss = torch.mean(mf_loss)

        # calculate L2 reg
        if self.reg_weight > 0.:
            user_embeddings = self.get_all_user_embedding()
            item_embeddings = self.get_all_item_embedding()
            u_ego_embeddings = user_embeddings[user]
            pos_ego_embeddings = item_embeddings[pos_item]
            neg_ego_embeddings = item_embeddings[neg_item]
            reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings).squeeze()
            loss += self.reg_weight * reg_loss

        # calculate agreement
        if self.MIM_weight > 0.:
            aug_user_all_embeddings, _ = self.ssl_forward()
            aug_u_embeddings = aug_user_all_embeddings[user]
            mutual_info = self.ssl_triple_loss(u_embeddings, aug_u_embeddings, aug_user_all_embeddings)
            loss += self.MIM_weight * mutual_info

        return loss

    # def predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #
    #     user_all_embeddings, item_all_embeddings = self.forward(pruning=self.prune_threshold)
    #
    #     u_embeddings = user_all_embeddings[user]
    #     i_embeddings = item_all_embeddings[item]
    #     scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
    #     return scores
    #
    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     if self.restore_user_e is None or self.restore_item_e is None:
    #         self.restore_user_e, self.restore_item_e = self.forward(pruning=self.prune_threshold)
    #
    #     self.restore_user_e, self.restore_item_e = self.forward(pruning=self.prune_threshold)
    #
    #     u_embeddings = self.restore_user_e[user]
    #     scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
    #     return scores.view(-1)





class RGCF_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "RGCF_RecommenderWrapper"

    def __init__(self, URM_train, use_gpu = False, verbose = True, use_cython_sampler = True):
        super(RGCF_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")


    def fit(self, epochs = None,
            GNN_layers_K = None,
            batch_size = None,
            embedding_size = None,
            prune_threshold_beta = None,
            contrastive_loss_temperature_tau = None,
            contrastive_loss_weight = None,
            augmentation_ratio = None,
            l2_reg = None,
            sgd_mode = None,
            learning_rate = None,
            **earlystopping_kwargs):

        self.prune_threshold_beta = prune_threshold_beta

        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        torch.cuda.empty_cache()

        self._model = RGCF(
             interaction_matrix = self.URM_train,
             embedding_size = embedding_size,
             n_layers = GNN_layers_K,
             reg_weight = l2_reg,
             prune_threshold = prune_threshold_beta,
             MIM_weight = contrastive_loss_weight,
             tau = contrastive_loss_temperature_tau,
             aug_ratio = augmentation_ratio,
             device = self.device,
             ).to(self.device)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, 0.0)

        ###############################################################################
        ### This is a standard training with early stopping part

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()
        self._model_state = self._model_state_best



    def _prepare_model_for_validation(self):
        with torch.no_grad():
            self.USER_factors, self.ITEM_factors = self._model.forward(pruning=self.prune_threshold_beta)
            self.USER_factors = self.USER_factors.detach().cpu().numpy()
            self.ITEM_factors = self.ITEM_factors.detach().cpu().numpy()
            self._model_state = clone_pytorch_model_to_numpy_dict(self._model)



    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self._model_state_best = copy.deepcopy(self._model_state)


    def _run_epoch(self, currentEpoch):

        if self.verbose:
            batch_iterator = tqdm(self._data_iterator)
        else:
            batch_iterator = self._data_iterator

        epoch_loss = 0
        for batch in batch_iterator:

            # Clear previously computed gradients
            self._optimizer.zero_grad()

            batch_loss = self._model.calculate_loss(batch, epoch_idx=currentEpoch)

            # Compute gradients given current loss
            batch_loss.backward()
            epoch_loss += batch_loss.item()

            # Apply gradient using the selected optimizer
            self._optimizer.step()

        self._print("Loss {:.2E}".format(epoch_loss))



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"USER_factors": self.USER_factors,
                             "ITEM_factors": self.ITEM_factors,
                             "_model_state": self._model_state,
                             }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")
