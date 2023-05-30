#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO

import pandas as pd
import numpy as np
from tqdm import tqdm
import math, torch, copy
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sps

from Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict

from SIGIR2022.HAKG_github.modules.HAKG import GraphConv
from SIGIR2022.HAKG_github.modules.hyperbolic import expmap0, sq_hyp_distance
# from SIGIR2022.HAKG_github.utils.parser import parse_args
# from SIGIR2022.HAKG_github.main import get_feed_data
# from SIGIR2022.HAKG_github.utils.evaluate import test
#
# import networkx as nx
# from collections import defaultdict
#
# def build_graph(train_data, triplets):
#     ckg_graph = nx.MultiDiGraph()
#     rd = defaultdict(list)
#
#     train_data = sps.coo_matrix(train_data)
#
#     print("Begin to load interaction triples ...")
#     for i in range(train_data.nnz):
#     # for u_id, i_id in tqdm(train_data, ascii=True):
#         u_id = train_data.row[i]
#         i_id = train_data.col[i]
#         rd[0].append([u_id, i_id])
#
#     print("\nBegin to load knowledge graph triples ...")
#     for index in tqdm(range(len(triplets))):
#     #for h_id, r_id, t_id in tqdm(triplets, ascii=True):
#         h_id = triplets.loc[index, "head"]
#         r_id = triplets.loc[index, "relation"]
#         t_id = triplets.loc[index, "tail"]
#         ckg_graph.add_edge(h_id, t_id, key=r_id)
#         rd[r_id].append([h_id, t_id])
#
#     return ckg_graph, rd
#
#
# def build_sparse_relational_graph(relation_dict, n_users, n_nodes):
#     def _bi_norm_lap(adj):
#         # D^{-1/2}AD^{-1/2}
#         rowsum = np.array(adj.sum(1))
#
#         d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#         d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#         d_mat_inv_sqrt = sps.diags(d_inv_sqrt)
#
#         # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
#         bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
#         return bi_lap.tocoo()
#
#     def _si_norm_lap(adj):
#         # D^{-1}A
#         rowsum = np.array(adj.sum(1))
#
#         d_inv = np.power(rowsum, -1).flatten()
#         d_inv[np.isinf(d_inv)] = 0.
#         d_mat_inv = sps.diags(d_inv)
#
#         norm_adj = d_mat_inv.dot(adj)
#         return norm_adj.tocoo()
#
#     adj_mat_list = []
#     print("Begin to build sparse relation matrix ...")
#     for r_id in tqdm(relation_dict.keys()):
#         np_mat = np.array(relation_dict[r_id])
#         if r_id == 0:
#             cf = np_mat.copy()
#             cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
#             vals = [1.] * len(cf)
#             adj = sps.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
#         else:
#             vals = [1.] * len(np_mat)
#             adj = sps.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
#         adj_mat_list.append(adj)
#
#     # norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
#     # mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
#     mean_mat_list = _si_norm_lap(adj_mat_list[0])
#     # interaction: user->item, [n_users, n_entities]
#     # norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
#     mean_mat_list = mean_mat_list.tocsr()[:n_users, n_users:].tocoo()
#
#     return adj_mat_list, mean_mat_list




class HAKGModel(nn.Module):
    def __init__(self, URM_train, knowledge_base_df,
                 margin_ccl, num_neg_sample, l2, angle_loss_w, emb_size,
                 context_hops, node_dropout_rate, mess_dropout_rate,
                 loss_f, device, angle_dropout_rate):
        super(HAKGModel, self).__init__()

        # self.n_users = data_config['n_users']
        # self.n_items = data_config['n_items']
        # self.n_relations = data_config['n_relations']
        # self.n_entities = data_config['n_entities']  # include items
        # self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        #
        # self.margin_ccl = args_config.margin
        # self.num_neg_sample = args_config.num_neg_sample
        #
        # self.decay = args_config.l2
        # self.angle_loss_w = args_config.angle_loss_w
        # self.emb_size = args_config.dim
        # self.context_hops = args_config.context_hops
        # self.node_dropout = args_config.node_dropout
        # self.node_dropout_rate = args_config.node_dropout_rate
        # self.mess_dropout = args_config.mess_dropout
        # self.mess_dropout_rate = args_config.mess_dropout_rate
        # self.loss_f = args_config.loss_f
        # self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
        #                                                               else torch.device("cpu")


        # The original implementation assumes Relation 0 is the interaction so all other relation IDs should be +1
        knowledge_base_df["relation"] += 1

        self.n_users, self.n_items = URM_train.shape

        # Entities are Items + Features
        self.n_entities = max(knowledge_base_df["head"].max(), knowledge_base_df["tail"].max()) +1
        self.n_relations = knowledge_base_df["relation"].max() +1
        # Nodes are Entities (Items + Features) + Users
        self.n_nodes = self.n_entities + self.n_users


        # Mean matrix = D^(-1)*A
        # A = |U|x|Entities| = |U|x(|I| + |Features|)
        # Note that A is computed only on collaborative data so the only non-zero entries
        # will be for the items and not for the features
        A = URM_train.copy()
        D_inv = 1/(np.sqrt(np.array(A.sum(axis = 1)).squeeze()) + 1e-6)
        A_tilde = sps.diags(D_inv).dot(A).astype(np.float32)
        A_tilde.resize((self.n_users, self.n_entities))
        A_tilde = sps.coo_matrix(A_tilde)


        # self.n_relations = n_relations
        # self.n_entities = n_entities  # include items
        # self.n_nodes = n_nodes  # n_users + n_entities

        self.margin_ccl = margin_ccl
        self.num_neg_sample = num_neg_sample

        self.decay = l2
        self.angle_loss_w = angle_loss_w
        self.emb_size = emb_size
        self.context_hops = context_hops
        self.node_dropout = node_dropout_rate > 0.0
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout = mess_dropout_rate > 0.0
        self.mess_dropout_rate = mess_dropout_rate
        self.loss_f = loss_f
        self.device = device

        self.angle_dropout_rate = angle_dropout_rate
        self.angle_emb_dropout = nn.Dropout(p=self.angle_dropout_rate)
        self.adj_mat = A_tilde

        # This edges are only referred to the entities and do not contain user-item interactions
        self.edge_index, self.edge_type = self._get_edges(knowledge_base_df)
        self.triplet_item_att = self._triplet_sampling(self.edge_index, self.edge_type)

        self._init_weight()
        self._init_loss_function()

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)
        self.item_emb_cf = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_items=self.n_items,
                         interact_mat=self.interact_mat,
                         device=self.device,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape, dtype=torch.float)
        # return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_edges(self, knowledge_base_df):
        # graph_tensor = torch.tensor(list(graph.edges))   # [-1, 3]
        # index = graph_tensor[:, :-1]  # [-1, 2]
        # type = graph_tensor[:, -1]  # [-1, 1]
        # return index.t().long().to(self.device), type.long().to(self.device)

        n_graph_edges = len(knowledge_base_df)

        index = torch.empty((2, n_graph_edges), dtype=torch.long, device=self.device)
        type = torch.empty((n_graph_edges,), dtype=torch.long, device=self.device)

        index[0,:] = torch.from_numpy(knowledge_base_df["head"].values)
        index[1,:] = torch.from_numpy(knowledge_base_df["tail"].values)
        type[:] = torch.from_numpy(knowledge_base_df["relation"].values)

        return index, type



    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == "dis_bpr":
            self.loss = self.create_dis_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def _triplet_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        # edge_index_t = edge_index.t()
        # sample = []
        # for idx, h_t in enumerate(edge_index_t):
        #     if (h_t[0]>=self.n_items and h_t[1]<self.n_items) or (h_t[0]<self.n_items and h_t[1]>=self.n_items):
        #         sample.append(idx)
        # sample = torch.LongTensor(sample)
        # return edge_index_t[sample]
        cross_type_edge = torch.logical_or(
            torch.logical_and(edge_index[0,:]>=self.n_items, edge_index[1,:]<self.n_items),
            torch.logical_and(edge_index[0,:]<self.n_items, edge_index[1,:]>=self.n_items))

        return edge_index[:,cross_type_edge]


    def half_aperture(self, u):
        eps = 1e-6
        K = 0.1
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return torch.asin((K * (1 - sqnu) / torch.sqrt(sqnu)).clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        eps = 1e-6
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norm_v ** 2))
        denom = (norm_u * edist * ((1 + norm_v ** 2 * norm_u ** 2 - 2 * dot_prod).clamp(min=eps).sqrt())) + eps
        return (num / denom).clamp_(min=-1 + eps, max=1 - eps).acos()

    def angle_loss(self, entity_emb, user):
        hier_hs = entity_emb[self.triplet_item_att[0]]
        hier_ts = entity_emb[self.triplet_item_att[1]]

        emb_drop = self.angle_emb_dropout(torch.ones(size=hier_hs.shape)) * self.angle_dropout_rate  # need to tune
        emb_drop = emb_drop.to(self.device)
        hier_hs = hier_hs * emb_drop
        hier_ts = hier_ts * emb_drop

        loss3 = 0
        batch_size = user.shape[0]
        num = self.triplet_item_att.shape[1]
        num_x = math.ceil(num / batch_size)
        for i in range(num_x):
            hier_h = hier_hs[i * batch_size:(i + 1) * batch_size]
            hier_t = hier_ts[i * batch_size:(i + 1) * batch_size]
            angle_half = self.angle_at_u(hier_h, hier_t) - self.half_aperture(hier_h)
            angle_half[angle_half < 0] = 0
            loss3 += torch.sum(angle_half)

        loss3 = self.angle_loss_w * loss3 / num

        return loss3


    def forward(self, user, pos_item, neg_item):
        # user = batch['users']
        # pos_item = batch['pos_items']
        # neg_item = batch['neg_items'].view(-1)
        neg_item = neg_item.view(-1)

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf = self.gcn(user_emb,
                                                     entity_emb,
                                                     self.item_emb_cf,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        pos_e_cf, neg_e_cf = item_gcn_emb_cf[pos_item], item_gcn_emb_cf[neg_item]
        loss1 = self.loss(u_e, pos_e, neg_e, pos_e_cf, neg_e_cf)
        loss2 = self.angle_loss(entity_emb, user)

        loss = loss1 + loss2

        return loss

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf = self.gcn(user_emb,
                                                    entity_emb,
                                                    self.item_emb_cf,
                                                    self.edge_index,
                                                    self.edge_type,
                                                    self.interact_mat,
                                                    mess_dropout=False, node_dropout=False)

        entity_gcn_emb[:self.n_items] += item_gcn_emb_cf
        return entity_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        if self.loss_f == "inner_bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        elif self.loss_f == 'contrastive_loss':
            # u_g_embeddings = F.normalize(u_g_embeddings)
            # i_g_embeddings = F.normalize(i_g_embeddings)
            return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2).detach().cpu()

        else:
            n_user = len(u_g_embeddings)
            n_item = len(i_g_embeddings)
            hyper_rate_matrix = np.zeros(shape=(n_user, n_item))

            hyper_u_g_embeddings = expmap0(u_g_embeddings)
            hyper_i_g_embeddings = expmap0(i_g_embeddings)

            for i in range(n_user):
                # [1, dim]
                one_hyper_u = hyper_u_g_embeddings[i, :]
                # [n_item, dim]
                one_hyper_u = one_hyper_u.expand(n_item, -1)
                one_hyper_score = -1 * sq_hyp_distance(one_hyper_u, hyper_i_g_embeddings)
                hyper_rate_matrix[i, :] = one_hyper_score.squeeze().detach().cpu()

            return hyper_rate_matrix

    def create_contrastive_loss(self, u_e, pos_e, neg_e, pos_e_cf, neg_e_cf):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)
        pos_e_cf = F.normalize(pos_e_cf)
        neg_e_cf = F.normalize(neg_e_cf)

        ui_pos = torch.relu(2 - (torch.cosine_similarity(u_e, pos_e, dim=1) + torch.cosine_similarity(u_e, pos_e_cf, dim=1)))
        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1>0
        ui_neg_loss1 = torch.sum(ui_neg1,dim=-1)/(torch.sum(x, dim=-1) + 1e-5)

        ui_neg2 = torch.relu(torch.cosine_similarity(users_batch, neg_e_cf, dim=1) - self.margin_ccl)
        ui_neg2 = ui_neg2.view(batch_size, -1)
        x = ui_neg2 > 0
        ui_neg_loss2 = torch.sum(ui_neg2, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos + ui_neg_loss1 + ui_neg_loss2

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss

    def create_dis_bpr_loss(self, users, pos_items, neg_items):
        hyper_users = expmap0(users)
        hyper_pos_items = expmap0(pos_items)
        hyper_neg_items = expmap0(neg_items)

        hyper_pos_dis = sq_hyp_distance(hyper_users, hyper_pos_items)
        hyper_neg_dis = sq_hyp_distance(hyper_users, hyper_neg_items)
        # hyper_pos_dis = hyp_distance(hyper_users, hyper_pos_items)
        # hyper_neg_dis = hyp_distance(hyper_users, hyper_neg_items)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()
                                  (hyper_neg_dis - hyper_pos_dis))
        return cf_loss




class HAKG_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "HAKG_RecommenderWrapper"

    def __init__(self, URM_train, knowledge_base_df, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(HAKG_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        self.knowledge_base_df = knowledge_base_df.copy()
        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")


    #
    #
    # def _compute_item_score(self, user_id_array, items_to_compute=None):
    #
    #     # Create the full data structure that will contain the item scores
    #     item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
    #
    #     if items_to_compute is not None:
    #         item_indices = items_to_compute
    #     else:
    #         item_indices = np.arange(self.n_items, dtype=np.int)
    #
    #     self._model.eval()
    #
    #     # entity_gcn_emb, user_gcn_emb = self._model.generate()
    #
    #     with torch.no_grad():
    #         user_batch = torch.LongTensor(np.array(user_id_array)).to(self.device)
    #         u_g_embeddings = self.USER_factors_tensor[user_batch].detach()
    #         i_g_embeddings = self.ITEM_factors_tensor[item_indices].detach()
    #         # i_rate_batch = torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2).detach().cpu().numpy()
    #         # i_rate_batch = self._model.rating(u_g_embeddings, i_g_embddings)
    #
    #     for user_index in range(len(user_batch)):
    #
    #         scores_for_user = torch.cosine_similarity(u_g_embeddings[user_index,:], i_g_embeddings, dim=1).cpu().numpy()
    #         dot_produt_user = self.USER_factors[user_id_array[user_index],:].dot(self.ITEM_factors.T)
    #
    #         try:
    #             assert np.allclose(scores_for_user, dot_produt_user)
    #         except:
    #             pass
    #
    #         if items_to_compute is not None:
    #             item_scores[user_index, items_to_compute] = scores_for_user[items_to_compute]
    #         else:
    #             item_scores[user_index, :] = scores_for_user
    #
    #     return item_scores
    #

    # def _init_model(self):
    #     """
    #     This function instantiates the model, it should only rely on attributes and not function parameters
    #     It should be used both in the fit function and in the load_model function
    #     :return:
    #     """
    #
    #     model = HAKGModel(n_params, self._params, graph, mean_mat_list).to(self._params.device)
    #     self._params.opt = torch.optim.Adam(model.parameters(), lr=self._params.lr)


    def fit(self,
            epochs = None,
            batch_size = None,
            embedding_size = None,
            sgd_mode = None,
            l2_reg = None,
            learning_rate = None,
            angle_loss_weight = None,
            contrastive_loss_margin = None,
            GNN_layers_K = None,
            add_inverse_relation = None,
            node_dropout_rate = None,
            mess_dropout_rate = None,
            angle_dropout_rate = None,
            n_negative_samples_M = None,
            **earlystopping_kwargs
            ):


        if add_inverse_relation:
            knowledge_base_inverse = self.knowledge_base_df.copy()
            knowledge_base_inverse.columns = ["tail", "relation", "head"]
            knowledge_base_inverse["relation"] += knowledge_base_inverse["relation"].max() + 1

            knowledge_base_df_training = pd.concat([self.knowledge_base_df, knowledge_base_inverse], ignore_index=True)
        else:
            knowledge_base_df_training = self.knowledge_base_df

        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size,
                                                        n_negatives_per_positive = n_negative_samples_M)

        torch.cuda.empty_cache()

        # graph, relation_dict = build_graph(self.URM_train, knowledge_base_df_training)
        #
        # print('building the adj mat ...')
        # _, mean_mat = build_sparse_relational_graph(relation_dict, self.n_users, len(graph.nodes))

        self._model = HAKGModel(URM_train = self.URM_train,
                                knowledge_base_df = knowledge_base_df_training,
                                margin_ccl = contrastive_loss_margin,
                                num_neg_sample = n_negative_samples_M,
                                l2 = l2_reg,
                                angle_loss_w = angle_loss_weight,
                                emb_size = embedding_size,
                                context_hops = GNN_layers_K,
                                node_dropout_rate = node_dropout_rate,
                                mess_dropout_rate = mess_dropout_rate,
                                loss_f = 'contrastive_loss',
                                device = self.device,
                                angle_dropout_rate = angle_dropout_rate
                                ).to(self.device)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, 0.0)

        ###############################################################################
        ### This is a standard training with early stopping part

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
            # self.ITEM_factors, self.USER_factors = self._model.generate()
            # self.ITEM_factors = self.ITEM_factors[:self.n_items,:]

            ITEM_factors, USER_factors = self._model.generate()
            ITEM_factors = ITEM_factors[:self.n_items,:].detach()
            USER_factors = USER_factors.detach()

            # self.ITEM_factors_tensor = ITEM_factors.clone()
            # self.USER_factors_tensor = USER_factors.clone()

            USER_factors_2_norm = torch.linalg.vector_norm(USER_factors, ord=2, dim=1)
            USER_factors = torch.div(USER_factors.T, USER_factors_2_norm[:None]).T

            # The item factors contain the item embeddings followed by the entity embeddings
            ITEM_factors_2_norm = torch.linalg.vector_norm(ITEM_factors, ord=2, dim=1)
            ITEM_factors = torch.div(ITEM_factors.T, ITEM_factors_2_norm[:None]).T

            self.USER_factors = USER_factors.cpu().numpy()
            self.ITEM_factors = ITEM_factors.cpu().numpy()

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

            user_batch, pos_item_batch, neg_item_batch = batch
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            loss = self._model(user_batch, pos_item_batch, neg_item_batch)

            # Compute gradients given current loss
            loss.backward()
            epoch_loss += loss.item()

            # Apply gradient using the selected _optimizer
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
