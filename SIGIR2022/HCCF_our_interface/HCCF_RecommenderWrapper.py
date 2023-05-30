#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO

from tqdm import tqdm
import torch
import scipy.sparse as sps
import copy

from normalized_adjacency_matrix import normalized_adjacency_matrix

from Utils.PyTorch.utils import get_optimizer, _sps_to_coo_tensor, clone_pytorch_model_to_numpy_dict
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator

import torch as t
from torch import nn
import torch.nn.functional as F
from SIGIR2022.HCCF_github.torchVersion.Utils.Utils import pairPredict, contrastLoss

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

class HCCFModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, hyperedge_size, GNN_layers_K, HYP_layers_C, temperature, leaky_relu_slope):
        super(HCCFModel, self).__init__()

        self.n_users = n_users
        self.GNN_layers_K = GNN_layers_K
        self.temperature = temperature

        # These are the embeddings E
        self.uEmbeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(n_users, embedding_size)))
        self.iEmbeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(n_items, embedding_size)))
        self.gcnLayer = GCNLayer(leaky_relu_slope)
        self.hgnnLayer = HGNNLayer(leaky_relu_slope, HYP_layers_C, hyperedge_size)

        # These are the low dimensional approximation matrices for the hypergraph adjacency matrix H = WE
        self.uHyper = nn.Parameter(nn.init.xavier_uniform_(t.empty(embedding_size, hyperedge_size)))
        self.iHyper = nn.Parameter(nn.init.xavier_uniform_(t.empty(embedding_size, hyperedge_size)))

        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        gnnLats = []
        hyperLats = []
        uuHyper = self.uEmbeds.mm(self.uHyper)
        iiHyper = self.iEmbeds.mm(self.iHyper)

        for i in range(self.GNN_layers_K):
            temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
            hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keepRate), lats[-1][:self.n_users])
            hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keepRate), lats[-1][self.n_users:])
            gnnLats.append(temEmbeds)
            hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
            lats.append(temEmbeds + hyperLats[-1])
        embeds = sum(lats)
        return embeds, gnnLats, hyperLats

    def calcLosses(self, ancs, poss, negs, adj, keepRate):
        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
        uEmbeds, iEmbeds = embeds[:self.n_users], embeds[self.n_users:]

        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().mean()
        # bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

        sslLoss = 0
        for i in range(self.GNN_layers_K):
            embeds1 = gcnEmbedsLst[i]#.detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:self.n_users], embeds2[:self.n_users], t.unique(ancs), self.temperature) + contrastLoss(embeds1[self.n_users:], embeds2[self.n_users:], t.unique(poss), self.temperature)
        return bprLoss, sslLoss

    def predict(self, adj):
        embeds, _, _ = self.forward(adj, 1.0)
        return embeds[:self.n_users], embeds[self.n_users:]

class GCNLayer(nn.Module):
    def __init__(self, leaky_relu_slope):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope)

    def forward(self, adj, embeds):
        return self.act(t.spmm(adj, embeds))

class HGNNLayer(nn.Module):
    def __init__(self, leaky_relu_slope, HYP_layers_C, hyperedge_size):
        super(HGNNLayer, self).__init__()
        self.HYP_layers_C = HYP_layers_C
        self.act = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.V = nn.Parameter(nn.init.xavier_uniform_(t.empty(hyperedge_size, hyperedge_size)))

    # def forward(self, adj, embeds):
    #     lat = self.act(adj.T.mm(embeds))
    #     ret = self.act(adj.mm(lat))
    #     return ret

    def forward(self, adj, embeds):
        lat = self.act(adj.T.mm(embeds))

        # c layers of hypergraph mapping
        # psi(x) = sigma(VX) + X
        for c in range(0, self.HYP_layers_C):
            lat = self.act(self.V.mm(lat)) + lat

        ret = self.act(adj.mm(lat))
        return ret

class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class HCCF_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "HCCF_RecommenderWrapper"

    def __init__(self, URM_train, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(HCCF_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")


    def fit(self, epochs = None,
            sgd_mode = None,
            learning_rate = None,
            embedding_size = None,
            GNN_layers_K = None,
            HYP_layers_C = None,
            hyperedge_size = None,
            batch_size = None,
            dropout = None,
            contrastive_loss_weight = None,
            l2_reg = None,
            learning_rate_decay = None,
            contrastive_loss_temperature_tau = None,
            leaky_relu_slope = None,

            # These are standard
            **earlystopping_kwargs
            ):

        self.l2_reg = l2_reg
        self.dropout = dropout
        self.contrastive_loss_weight = contrastive_loss_weight

        A_tilde = normalized_adjacency_matrix(self.URM_train, add_self_connection = True)
        A_tilde = sps.csr_matrix(A_tilde)

        torch.cuda.empty_cache()

        self._model = HCCFModel(n_users = self.n_users,
                                n_items = self.n_items,
                                embedding_size = embedding_size,
                                hyperedge_size = hyperedge_size,
                                GNN_layers_K = GNN_layers_K,
                                HYP_layers_C = HYP_layers_C,
                                temperature = contrastive_loss_temperature_tau,
                                leaky_relu_slope = leaky_relu_slope).to(self.device)

        self.A_tilde = _sps_to_coo_tensor(A_tilde, self.device)

        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, 0.0)
        self._learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = self._optimizer, gamma = learning_rate_decay)

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
            self.USER_factors, self.ITEM_factors = self._model.predict(self.A_tilde)
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

            user_batch, pos_item_batch, neg_item_batch = batch
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            bprLoss, sslLoss = self._model.calcLosses(user_batch, pos_item_batch, neg_item_batch, self.A_tilde, 1 - self.dropout)
            sslLoss = sslLoss * self.contrastive_loss_weight
            
            regLoss = calcRegLoss(self._model) * self.l2_reg
            batch_loss = bprLoss + sslLoss + regLoss

            # Compute gradients given current loss
            batch_loss.backward()
            epoch_loss += batch_loss.item()

            # Apply gradient using the selected optimizer
            self._optimizer.step()

        # Apply decay of learning rate at the end of each epoch
        self._learning_rate_scheduler.step()

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
