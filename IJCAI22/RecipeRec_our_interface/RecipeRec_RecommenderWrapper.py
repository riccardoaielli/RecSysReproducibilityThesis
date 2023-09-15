#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/08/2023

@author: Riccardo Luigi Aielli
"""
import json
import pickle
import re
import nltk
from collections import Counter
import pandas as pd
import random
import heapq
import csv
from tqdm import tqdm
import os
import numpy as np
import time
import math
# import lmdb
import gensim
import heapq

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
import torchfile
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
# from dgl.nn import EdgeWeightNorm

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO

import gc
import numpy as np
import torch
import scipy.sparse as sps

from Utils.PyTorch.utils import get_optimizer
from IJCAI22.RecipeRec_our_interface.RecipeRec_our_interface import *


class RecipeRec_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "RecipeRec_RecommenderWrapper"

    def __init__(self, URM_train, ICM_train, graph, train_graph, val_graph, train_edgeloader, val_edgeloader, test_edgeloader, verbose=True, use_gpu=True):
        # TODO remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        super(RecipeRec_RecommenderWrapper, self).__init__(
            URM_train, ICM_train, verbose=verbose)

        self.graph = graph
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.train_edgeloader = train_edgeloader
        self.val_edgeloader = val_edgeloader
        self.test_edgeloader = test_edgeloader

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                self.device = "mps"
        else:
            print("GPU is not available, using cpu")
            self.device = torch.device("cpu:0")

        print('device: ', self.device)

        self.temp_file_folder = None

        self.warm_user_ids = np.arange(0, self.n_users)[np.ediff1d(
            sps.csr_matrix(self.URM_train).indptr) > 0]
        # self.warm_user_ids = torch.from_numpy(self.warm_user_ids).to(self.device)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # In order to compute the prediction the model may need a Session. The session is an attribute of this Wrapper.
        # There are two possible scenarios for the creation of the session: at the beginning of the fit function (training phase)
        # or at the end of the fit function (before loading the best model, testing phase)

        # predict_all = self.model.predict_matrix().detach().cpu().numpy()
        predict_batch = self._model.predict_batch(
            user_id_array).detach().cpu().numpy()

        if items_to_compute is not None:
            item_scores = - \
                np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = predict_batch[:,
                                                             items_to_compute]
        else:
            item_scores = predict_batch

        return item_scores

    def _init_model(self, spectral_features_dict=None):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        # get ingre neighbors for each recipe nodes

        def get_recipe2ingreNeighbor_dict():
            max_length = 33
            out = {}
            neighbor_list = []
            ingre_length_list = []
            total_length_index_list = []
            total_ingre_neighbor_list = []
            total_length_index = 0
            total_length_index_list.append(total_length_index)
            for recipeNodeID in tqdm(range(self.graph.number_of_nodes('recipe'))):
                _, succs = self.graph.out_edges(recipeNodeID, etype='r-i')
                succs_list = list(set(succs.tolist()))
                total_ingre_neighbor_list.extend(succs_list)
                cur_length = len(succs_list)
                ingre_length_list.append(cur_length)

                total_length_index += cur_length
                total_length_index_list.append(total_length_index)
                while len(succs_list) < max_length:
                    succs_list.append(77733)
                neighbor_list.append(succs_list)

            ingre_neighbor_tensor = torch.tensor(neighbor_list)
            ingre_length_tensor = torch.tensor(ingre_length_list)
            total_ingre_neighbor_tensor = torch.tensor(
                total_ingre_neighbor_list)
            return ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, total_ingre_neighbor_tensor

        self.ingre_neighbor_tensor, self.ingre_length_tensor, self.total_length_index_list, self.total_ingre_neighbor_tensor = get_recipe2ingreNeighbor_dict()
        print('ingre_neighbor_tensor: ', self.ingre_neighbor_tensor.shape)
        print('ingre_length_tensor: ', self.ingre_length_tensor.shape)
        print('total_length_index_list: ', len(self.total_length_index_list))
        print('total_ingre_neighbor_tensor: ',
              self.total_ingre_neighbor_tensor.shape)

        # Passo il modello da trainare, non credo sia giusto farlo così
        self._model = RecipeRec(
            device=self.device,
            ingre_neighbor_tensor=self.ingre_neighbor_tensor,
            ingre_length_tensor=self.ingre_length_tensor,
            total_length_index_list=self.total_length_index_list,
            total_ingre_neighbor_tensor=self.total_ingre_neighbor_tensor,
            graph=self.graph,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            drop_out=self.drop_out,
            embedding_size=self.embedding_size,
            reg=self.reg,
            temperature=self.temperature,
            attentions_heads=self.attentions_heads,
            gamma=self.gamma,
        ).to(self.device)

    def fit(self,
            epochs=None,
            batch_size=None,
            learning_rate=None,
            temperature=None,
            attentions_heads=None,
            drop_out=None,
            embedding_size=None,
            reg=None,
            gamma=None,

            # These are standard
            **earlystopping_kwargs
            ):

        # set hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.embedding_size = embedding_size
        self.reg = reg
        self.temperature = temperature
        self.attentions_heads = attentions_heads
        self.gamma = gamma

        # TODO inizializza il modello
        self._init_model(spectral_features_dict=None)
        # Ottimizzatore
        opt = torch.optim.Adam(self._model.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, self.gamma)

        ###############################################################################
        # This is a standard training with early stopping part, most likely you won't need to change it

        gc.collect()
        torch.cuda.empty_cache()

        self._update_best_model()

        # Funzione che inizia ad eseguire le epoche e il training
        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # Colonano il modello per salvarne il modello migliore
        self._model.user_embed = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(self._USER_factors_best).to(self.device))
        self._model.item_embed = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(self._ITEM_factors_best).to(self.device))

        self._print("Training complete")

    def _prepare_model_for_validation(self):
        pass

    # TODO cambiare: va clonato l'oggetto pythorch e usato per ripristinare lo stato a quel modello finito il training.
    # Questo perchè devo usare il modello migliore per fare evaluation
    def _update_best_model(self):
        self._USER_factors_best = self._model.user_embed.weight.clone().detach().cpu().numpy()
        self._ITEM_factors_best = self._model.item_embed.weight.clone().detach().cpu().numpy()

    # contiene tutto il necessario per far runnare una singola epoca
    def _run_epoch(self, currentEpoch):
        # TODO replace this with the train loop for one epoch of the model

        train_start = time.time()
        epoch_loss = 0
        epoch_contrastive_loss = 0
        epoch_emb_loss = 0
        iteration_cnt = 0

        for input_nodes, positive_graph, negative_graph, blocks in self.train_edgeloader:
            self.model.train()
            blocks = [b.to(self.device) for b in blocks]
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)

            input_user = blocks[0].srcdata['random_feature']['user']
            input_instr = blocks[0].srcdata['avg_instr_feature']['recipe']
            input_ingredient = blocks[0].srcdata['nutrient_feature']['ingredient']
            ingredient_of_dst_recipe = blocks[1].srcdata['nutrient_feature']['ingredient']
            input_features = [input_user, input_instr,
                              input_ingredient, ingredient_of_dst_recipe]

            pos_score, neg_score, x1, x2 = self.model(
                positive_graph, negative_graph, blocks, input_features)
            contrastive_loss = get_contrastive_loss(x1, x2, self.temperature)
            # emb_loss = get_emb_loss(x1, x2)
            assert not math.isnan(contrastive_loss)
            recommendation_loss = get_recommendation_loss(pos_score, neg_score)
            assert not math.isnan(recommendation_loss)

            loss = recommendation_loss + self.drop_out * \
                contrastive_loss  # + 1e-5 * emb_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            epoch_loss += recommendation_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            # epoch_emb_loss += emb_loss.item()
            iteration_cnt += 1

            # break

        epoch_loss /= iteration_cnt
        epoch_contrastive_loss /= iteration_cnt
        train_end = time.strftime(
            "%M:%S min", time.gmtime(time.time()-train_start))

        print('Current Epoch: {0},  Loss: {l:.4f}, Contrastive: {cl:.4f}, Emb: {el:.4f},  Time: {t}, LR: {lr:.6f}'
              .format(currentEpoch, l=epoch_loss, cl=epoch_contrastive_loss, el=epoch_emb_loss, t=train_end, lr=self.opt.param_groups[0]['lr']))
        self.scheduler.step()

        # Evaluation
        # For demonstration purpose, only test set result is reported here. Please use val_dataloader for comprehensiveness.
        # if epoch >= 4 and epoch % 1 == 0:
        #     print('testing: ')
        #     evaluate(self.model, test_edgeloader, multi_metrics=True)
        #     print()
        #     print()

        """ n = self.ICM_train.shape[0]

        # for epoch in range(self._params.n_epochs):
        num_iter = int(n / self._params.batch_size)
        # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)
        gen_loss = self.model.cdl_estimate(self.ICM_train, num_iter)
        self.model.m_theta[:] = self.model.transform(self.ICM_train)
        likelihood = self.model.pmf_estimate(
            self._train_users, self._train_items, None, None, self._params)
        loss = -likelihood + 0.5 * gen_loss * n * self._params.lambda_r

        self.USER_factors = self.model.m_U.copy()
        self.ITEM_factors = self.model.m_V.copy() """

        logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
            currentEpoch, loss, -likelihood, gen_loss))

    # TODO Modifica
    # L'ideale sarebbe riuscire a clonare l'intero stato pythorch del modello con una funzione di libreria e nella load caruicarlo sempre allo stesso modo
    # evito di selezionare a mano i vari parametri e spostarli in giro
    # Gli oggetti pythorch sono dei grandi dizionari quindi volendo posso usare questa struttura per salvare il modello.
    # Approfondisci la funzione get_state_dictionary o get_parameter_dict sul modello pythorch che dovrebbe estrarre ciò che caratterizza il modello specifico, così evito quello che ha fatto il prof a mano
    # Per la load devo fare la stessa cosa al contrario
    # Praticamente mette tutto quello che serve per ripristinare quel modello in un dizionario che viene salvato in uno zip

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        spectral_features_dict = self._model.get_spectral_features()

        _USER_factors = self._model.user_embed.weight.clone().detach().cpu().numpy()
        _ITEM_factors = self._model.item_embed.weight.clone().detach().cpu().numpy()

        data_dict_to_save = {
            # Tutto ciò che serve per il costruttore del modello
            # TODO aggiungere gli Hyperparameters, forse non serve dipende come salvo il modello
            'drop_out': self.drop_out,
            'embedding_size': self.embedding_size,
            'reg': self.reg,
            'batch_size': self.batch_size,


            # Quello che serve per il modello
            # Model parameters
            "model_spectral_features_dict": spectral_features_dict,
            "model_USER_factors": _USER_factors,
            "model_ITEM_factors": _ITEM_factors,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name,
                         data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            if not attrib_name.startswith("model_"):
                self.__setattr__(attrib_name, data_dict[attrib_name])

        self._init_model(
            spectral_features_dict=data_dict["model_spectral_features_dict"])

        _USER_factors = data_dict["model_USER_factors"]
        _ITEM_factors = data_dict["model_ITEM_factors"]

        # Creo l'istanza del modello e inserisco gli iperparametri salvati
        self._model.user_embed = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(_USER_factors).to(self.device))
        self._model.item_embed = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(_ITEM_factors).to(self.device))

        self._print("Loading complete")
