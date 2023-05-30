#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

import torch
import torch.nn as nn
from torch.nn.init import normal_
import torch.nn.functional as F
from Recommenders.DataIO import DataIO
import copy
from tqdm import tqdm
import numpy as np
import scipy.sparse as sps

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from normalized_adjacency_matrix import normalized_adjacency_matrix, from_sparse_to_tensor

from Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator

from sklearn.preprocessing import normalize
import networkx as nx

from SIGIR2022.INMO_github.utils import get_sparse_tensor, generate_daj_mat



class DoubleIterator(object):

    def __init__(self, iterator1, iterator2):
        super(DoubleIterator, self).__init__()

        self.iterator1 = iterator1
        self.iterator2 = iterator2

        assert len(self.iterator1) == len(self.iterator2)

    def __len__(self):
        return len(self.iterator1)

    def __iter__(self):
        self.n_sampled_points = 0
        self.iterator1 = self.iterator1.__iter__()
        self.iterator2 = self.iterator2.__iter__()

        return self

    def __next__(self):

        batch1 = self.iterator1.__next__()
        batch2 = self.iterator2.__next__()

        assert len(batch1) == len(batch2)

        return batch1, batch2






def graph_rank_nodes(n_users, n_items, adj_mat, ranking_metric):

    if ranking_metric == 'degree':
        user_metrics = np.array(np.sum(adj_mat[:n_users, :], axis=1)).squeeze()
        item_metrics = np.array(np.sum(adj_mat[n_users:, :], axis=1)).squeeze()
    elif ranking_metric == 'greedy' or ranking_metric == 'sort':
        '''
        # This is for theoretical analysis.
        part_adj = adj_mat[:dataset.n_users, dataset.n_users:]
        part_adj_tensor = get_sparse_tensor(part_adj, 'cpu')
        with torch.no_grad():
            u, s, v = torch.svd_lowrank(part_adj_tensor, 64)
            u, v = u.numpy(), v.numpy()
        user_metrics = greedy_or_sort(part_adj, u, ranking_metric, dataset.device)
        item_metrics = greedy_or_sort(part_adj.T, v, ranking_metric, dataset.device)
        '''
        normalized_adj_mat = normalize(adj_mat, axis=1, norm='l1')
        user_metrics = np.array(np.sum(normalized_adj_mat[:, :n_users], axis=0)).squeeze()
        item_metrics = np.array(np.sum(normalized_adj_mat[:, n_users:], axis=0)).squeeze()
    elif ranking_metric == 'page_rank':
        g = nx.Graph()
        g.add_edges_from(np.array(np.nonzero(adj_mat)).T)
        pr = nx.pagerank(g)
        pr = np.array([pr[i] for i in range(n_users + n_items)])
        user_metrics, item_metrics = pr[:n_users], pr[n_users:]
    else:
        return None
    ranked_users = np.argsort(user_metrics)[::-1].copy()
    ranked_items = np.argsort(item_metrics)[::-1].copy()
    return ranked_users, ranked_items


class IGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, A_normalized, n_layers, dropout, template_ratio, delta, template_node_ranking_metric, device):
        super(IGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.template_ratio = template_ratio
        self.device = device
        # self.norm_adj = self.generate_graph(model_config['dataset'])
        self.norm_adj = from_sparse_to_tensor(A_normalized).to(self.device)
        self.alpha = 1.
        self.delta = delta

        A_not_normalized = A_normalized.astype(np.bool)
        self.template_mat, self.core_users, self.core_items, self.row_sum = self.generate_feat(A_not_normalized, ranking_metric=template_node_ranking_metric)
        self.update_feat_mat()

        self.embedding = nn.Embedding(self.template_mat.shape[1], self.embedding_size, dtype=torch.float32, device=self.device)
        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)


    def update_feat_mat(self):
        row, _ = self.template_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.template_mat = torch.sparse.FloatTensor(self.template_mat.indices(), edge_values, self.template_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        # return LightGCN.generate_graph(self, dataset)
        adj_mat = generate_daj_mat(dataset)
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)
        d_mat = sps.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj

    def generate_feat(self, A_not_normalized, ranking_metric=None):

        if self.template_ratio < 1.:
            ranked_users, ranked_items = graph_rank_nodes(self.n_users, self.n_items, A_not_normalized, ranking_metric)
            core_users = ranked_users[:int(self.n_users * self.template_ratio)]
            core_items = ranked_items[:int(self.n_items * self.template_ratio)]
        else:
            core_users = np.arange(self.n_users, dtype=np.int64)
            core_items = np.arange(self.n_items, dtype=np.int64)

        # user_map = {user:index for index, user in enumerate(core_users)}
        # item_map = {item:index for index, item in enumerate(core_items)}

        node_select_index = np.concatenate([core_users, self.n_users + core_items])
        template_A_sparse = A_not_normalized[:,node_select_index]

        # Adding an edge for all users to the global user template and the same for the items to a global item template
        global_user_item_template = np.zeros((template_A_sparse.shape[0], 2))
        global_user_item_template[0:self.n_users,0] = 1
        global_user_item_template[self.n_users:,1] = 1

        template_A_sparse = sps.hstack([template_A_sparse, sps.csr_matrix(global_user_item_template)])

        # user_dim, item_dim = len(user_map), len(item_map)
        # indices = []
        #
        # URM_sparse = sps.coo_matrix(URM_sparse)
        #
        # for index in range(URM_sparse.nnz):
        #     user = URM_sparse.row[index]
        #     item = URM_sparse.col[index]
        #
        #     if user==0 and item==96436:
        #         pass
        #
        #     if item in item_map:
        #         indices.append([user, user_dim + item_map[item]])
        #         assert template_A_sparse[user, user_dim + item_map[item]]
        #     if user in user_map:
        #         indices.append([self.n_users + item, user_map[user]])
        #         assert template_A_sparse[self.n_users + item, user_map[user]]
        #
        # for user in range(self.n_users):
        #     indices.append([user, user_dim + item_dim])
        # for item in range(self.n_items):
        #     indices.append([self.n_users + item, user_dim + item_dim + 1])
        # feat = sps.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
        #                      shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        #
        # row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        # feat = get_sparse_tensor(feat, self.device)
        # return feat, user_map, item_map, row_sum


        row_sum = torch.tensor(np.array(template_A_sparse.sum(axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        template_A_sparse = get_sparse_tensor(template_A_sparse, self.device)
        return template_A_sparse, core_users, core_items, row_sum



    def inductive_rep_layer(self, feat_mat):
        # This is the original implementation
        # padding_tensor = torch.empty([max(self.template_mat.shape) - self.template_mat.shape[1], self.embedding_size],
        #                              dtype=torch.float32, device=self.device)
        # padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)
        #
        # row, column = feat_mat.indices()
        # g = dgl.graph((column, row), num_nodes=max(self.template_mat.shape), device=self.device)
        # x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        # x = x[:self.template_mat.shape[0], :]

        # This is the modified implementation
        x = torch.sparse.mm(feat_mat, self.embedding.weight)
        return x

    def dropout_sp_mat(self, mat):
        if not self.training:
            return mat
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(mat._nnz()).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = mat.indices()
        v = mat.values()

        i = i[:, dropout_mask]
        v = v[dropout_mask] / (1. - self.dropout)
        out = torch.sparse.FloatTensor(i, v, mat.shape).coalesce()
        return out

    def get_rep(self):
        # template_mat = NGCF.dropout_sp_mat(self, self.template_mat)
        feat_mat = self.dropout_sp_mat(self.template_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]

        # This is the original implementation with DGL
        # row, column = self.norm_adj.indices()
        # g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        # for _ in range(self.n_layers):
        #     representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
        #     all_layer_rep.append(representations)

        # This is the modified implementation with torch sparse mm
        for _ in range(self.n_layers):
            representations = torch.sparse.mm(self.norm_adj, representations)
            all_layer_rep.append(representations)
        # End of the modified implementation

        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_final_representation(self):
        representations = self.inductive_rep_layer(self.template_mat)

        all_layer_rep = [representations]
        for _ in range(self.n_layers):
            representations = torch.sparse.mm(self.norm_adj, representations)
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep, self.template_mat.detach(), self.embedding.weight.detach()

    def bpr_forward(self, users, pos_items, neg_items):
        # return NGCF.bpr_forward(self, users, pos_items, neg_items)
        rep = self.get_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq


    def predict(self, users):
        # return LightGCN.predict(self, users)
        rep = self.get_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    # def save(self, path):
    #     params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
    #               'item_map': self.item_map, 'alpha': self.alpha}
    #     torch.save(params, path)
    #
    # def load(self, path):
    #     params = torch.load(path, map_location=self.device)
    #     self.load_state_dict(params['sate_dict'])
    #     self.user_map = params['user_map']
    #     self.item_map = params['item_map']
    #     self.alpha = params['alpha']
    #     self.template_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
    #     self.update_feat_mat()

def _get_URM_train_template(URM_train, core_users, core_items):
    # URM_train = sps.coo_matrix(URM_train)
    #
    # row = [user_map[URM_train.row[index]] for index in range(URM_train.nnz) if URM_train.row[index] in user_map]
    # col = [item_map[URM_train.col[index]] for index in range(URM_train.nnz) if URM_train.col[index] in item_map]
    #
    # URM_train_template = sps.csr_matrix(
    #     ([True]*len(row),
    #      (row, col)),
    #     dtype = URM_train.dtype,
    #     shape = URM_train.shape
    # )

    URM_train_template = URM_train[core_users,:]
    URM_train_template = URM_train_template[:,core_items]

    return URM_train_template


class INMO_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "INMO_RecommenderWrapper"

    def __init__(self, URM_train, use_gpu = False, use_cython_sampler = True, verbose = True):
        super(INMO_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")


    def fit(self, epochs = None,
            K = None,
            batch_size = None,
            embedding_size = None,
            dropout = None,
            template_ratio = None,
            template_loss_weight = None,
            template_node_ranking_metric = None,
            normalization_decay = None,
            l2_reg = None,
            sgd_mode = None,
            learning_rate = None,
            **earlystopping_kwargs):

        self.l2_reg = l2_reg
        self.template_loss_weight = template_loss_weight

        A_normalized = normalized_adjacency_matrix(self.URM_train)

        torch.cuda.empty_cache()

        self._model = IGCN(n_users = self.n_users,
                           n_items = self.n_items,
                           embedding_size = embedding_size,
                           A_normalized = A_normalized,
                           n_layers = K,
                           dropout = dropout,
                           template_ratio = template_ratio,
                           delta = normalization_decay,
                           template_node_ranking_metric = template_node_ranking_metric,
                           device = self.device).to(self.device)

        data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        URM_train_template = _get_URM_train_template(self.URM_train, self._model.core_users, self._model.core_items)
        data_iterator_template = self._data_iterator_class(URM_train_template, batch_size = batch_size, set_n_samples_to_draw = self.URM_train.nnz)

        self._data_iterator = DoubleIterator(data_iterator, data_iterator_template)

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
            ALL_factors, _, _ = self._model.get_final_representation()
            ALL_factors = ALL_factors.cpu().numpy()
            self.USER_factors = ALL_factors[:self.n_users, :]
            self.ITEM_factors = ALL_factors[self.n_users:, :]
            self._model_state = clone_pytorch_model_to_numpy_dict(self._model)


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self._model_state_best = copy.deepcopy(self._model_state)


    def _run_epoch(self, num_epoch):
        
        if self.verbose:
            batch_iterator = tqdm(self._data_iterator)
        else:
            batch_iterator = self._data_iterator
        
        epoch_loss = 0
        for batch in batch_iterator:
            
            # Clear previously computed gradients
            self._optimizer.zero_grad()

            batch_original, batch_template = batch

            user_batch, pos_item_batch, neg_item_batch = batch_original
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            users_r, pos_items_r, neg_items_r, l2_norm_sq = self._model.bpr_forward(user_batch, pos_item_batch, neg_item_batch)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()


            user_batch, pos_item_batch, neg_item_batch = batch_template
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            users_r = self._model.embedding(user_batch)
            pos_items_r = self._model.embedding(pos_item_batch + len(self._model.core_users))
            neg_items_r = self._model.embedding(neg_item_batch + len(self._model.core_users))
            pos_scores = torch.sum(users_r * pos_items_r * self._model.w[None, :], dim=1)
            neg_scores = torch.sum(users_r * neg_items_r * self._model.w[None, :], dim=1)
            aux_loss = F.softplus(neg_scores - pos_scores).mean()

            batch_loss = bpr_loss + self.template_loss_weight * aux_loss + self.l2_reg * l2_norm_sq.mean()

            # Compute gradients given current loss
            batch_loss.backward()
            epoch_loss += batch_loss.item()
    
            # Apply gradient using the selected optimizer
            self._optimizer.step()

        self._model.feat_mat_anneal()

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
