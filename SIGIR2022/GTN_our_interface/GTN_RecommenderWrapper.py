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

from torch_sparse import SparseTensor
from SIGIR2022.GTN_github.code.gtn_propagation import GeneralPropagation
from types import SimpleNamespace

class GTN(nn.Module):
    def __init__(self, URM_train, embedding_size, GNN_layers_K, dropout_rate_lightgcn, dropout_rate_gtn, embedding_smoothness_weight, dual_stepsize_beta, device):
        super(GTN, self).__init__()
        # self.config = config
        # self.dataset: BasicDataset = dataset
        # self.args = args

        self.num_users, self.num_items = URM_train.shape
        self.latent_dim = embedding_size
        self.n_layers = GNN_layers_K

        self.dropout_rate_lightgcn = dropout_rate_lightgcn
        self.dropout = self.dropout_rate_lightgcn > 0.0

        self.A_split = False
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # if self.config['pretrain'] == 0:
        #     nn.init.normal_(self.embedding_user.weight, std=0.1)
        #     nn.init.normal_(self.embedding_item.weight, std=0.1)
        #     # world.cprint('use NORMAL distribution initilizer')
        # else:
        #     self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
        #     self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        #     print('use pretarined data')

        self.f = nn.Sigmoid()
        # self.Graph = self.dataset.getSparseGraph()

        URM_train = URM_train.astype(np.bool)
        Zero_u_u_sps = sps.csr_matrix((self.num_users, self.num_users))
        Zero_i_i_sps = sps.csr_matrix((self.num_items, self.num_items))
        A = sps.bmat([[Zero_u_u_sps,     URM_train ],
                      [URM_train.T, Zero_i_i_sps   ]], format="coo")

        self.Graph = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data,
                                             size=A.shape, dtype=torch.float32, device=device, requires_grad=False).coalesce()

        self.args = SimpleNamespace(prop_dropout = dropout_rate_gtn,
                               ogb = True,
                               incnorm_para = True,
                               lambda2 = embedding_smoothness_weight,
                               beta = dual_stepsize_beta,
                               debug = False)

        # print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # GeneralPropagation requires an alpha parameter which is never used
        self.gp = GeneralPropagation(self.n_layers, None, cached=True, args=self.args, device=device)


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g.coalesce()

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self, corrupted_graph=None):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if self.dropout:
            if self.training:
                g_droped = self.__dropout(1-self.dropout_rate_lightgcn)
            else:
                if corrupted_graph == None:
                    g_droped = self.Graph
                else:
                    g_droped = corrupted_graph
        else:
            if corrupted_graph == None:
                g_droped = self.Graph
            else:
                g_droped = corrupted_graph

        # our GCNs
        x = all_emb
        rc = g_droped.indices()
        r = rc[0]
        c = rc[1]
        num_nodes = g_droped.shape[0]
        edge_index = SparseTensor(row=r, col=c, value=g_droped.values(), sparse_sizes=(num_nodes, num_nodes))
        emb, embs = self.gp.forward(x, edge_index, mode="GTN")
        light_out = emb

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    # def getUsersRating(self, users):
    #     all_users, all_items = self.computer()
    #     users_emb = all_users[users.long()]
    #     items_emb = all_items
    #     rating = self.f(torch.matmul(users_emb, items_emb.t()))
    #     return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma




class GTN_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "GTN_RecommenderWrapper"

    def __init__(self, URM_train, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(GTN_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

    def fit(self,
            epochs = None,
            batch_size = None,
            embedding_size = None,
            sgd_mode = None,
            l2_reg = None,
            learning_rate = None,
            embedding_smoothness_weight = None,
            GNN_layers_K = None,
            dropout_rate_lightgcn = None,
            dropout_rate_gtn=None,
            **earlystopping_kwargs
            ):

        self.l2_reg = l2_reg
        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        torch.cuda.empty_cache()

        self._model = GTN(URM_train = self.URM_train,
                          embedding_size = embedding_size,
                          GNN_layers_K = GNN_layers_K,
                          dropout_rate_lightgcn = dropout_rate_lightgcn,
                          dropout_rate_gtn = dropout_rate_gtn,
                          embedding_smoothness_weight = embedding_smoothness_weight,
                          dual_stepsize_beta = 0.5,   # Value comes from the theoretical guarantee for convergence (gamma 1, beta 1/2)
                          device = self.device).to(self.device)

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
            self._model.eval()
            self.USER_factors, self.ITEM_factors = self._model.computer()

            self.USER_factors = self.USER_factors.detach().cpu().numpy()
            self.ITEM_factors = self.ITEM_factors.detach().cpu().numpy()

            self._model_state = clone_pytorch_model_to_numpy_dict(self._model)
            self._model.train()


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

            mf_loss, reg_loss = self._model.bpr_loss(user_batch, pos_item_batch, neg_item_batch)
            loss = mf_loss + reg_loss*self.l2_reg

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
