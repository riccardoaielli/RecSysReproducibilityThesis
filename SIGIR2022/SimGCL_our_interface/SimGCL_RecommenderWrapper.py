#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Recommenders.DataIO import DataIO
import copy

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from normalized_adjacency_matrix import normalized_adjacency_matrix
from Utils.PyTorch.utils import get_optimizer, clone_pytorch_model_to_numpy_dict
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator

from SIGIR2022.SimGCL_torch_github.base.torch_interface import TorchGraphInterface
from SIGIR2022.SimGCL_torch_github.util.loss_torch import bpr_loss, InfoNCE

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg
        
class SimGCL_Encoder(nn.Module):
    def __init__(self, n_users, n_items, adjacency_sparse, emb_size, eps, tau, n_layers, device):
        super(SimGCL_Encoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.eps = eps
        self.tau = tau
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = adjacency_sparse
        self.embedding_dict = self._init_model(device)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

    def _init_model(self, device):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size))),
        }).to(device)
        return embedding_dict

    def forward(self, perturbed=False):
        
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
                
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings


    def _calculate_CL_loss(self, u_idx, i_idx):
        # u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(self.device)
        # i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(self.device)
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.tau)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.tau)
        return user_cl_loss + item_cl_loss


class SimGCL_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "SimGCL_RecommenderWrapper"

    def __init__(self, URM_train, use_gpu = False, use_cython_sampler = True, verbose = True):
        super(SimGCL_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

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
            noise_magnitude_epsilon = None,
            contrastive_loss_temperature_tau = None,
            contrastive_loss_weight = None,
            l2_reg = None,
            sgd_mode = None,
            learning_rate = None,
            **earlystopping_kwargs):

        self.contrastive_loss_weight = contrastive_loss_weight
        self.l2_reg = l2_reg

        A_tilde = normalized_adjacency_matrix(self.URM_train)

        self._data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        torch.cuda.empty_cache()

        self._model = SimGCL_Encoder(self.n_users,
                                     self.n_items,
                                     A_tilde,
                                     embedding_size,
                                     noise_magnitude_epsilon,
                                     contrastive_loss_temperature_tau,
                                     GNN_layers_K, self.device).to(self.device)
        
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
            self.USER_factors, self.ITEM_factors = self._model()
            self.USER_factors = self.USER_factors.detach().cpu().numpy()
            self.ITEM_factors = self.ITEM_factors.detach().cpu().numpy()
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

            user_batch, pos_item_batch, neg_item_batch = batch
            USER_factors, ITEM_factors = self._model()
            
            rec_loss = bpr_loss(USER_factors[user_batch],
                                ITEM_factors[pos_item_batch],
                                ITEM_factors[neg_item_batch])
            
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            cl_loss = self.contrastive_loss_weight * self._model._calculate_CL_loss(user_batch, pos_item_batch)
            batch_loss =  rec_loss + cl_loss + l2_reg_loss(self.l2_reg, USER_factors[user_batch], ITEM_factors[pos_item_batch])
            
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

