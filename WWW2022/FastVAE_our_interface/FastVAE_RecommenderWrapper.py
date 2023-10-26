#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


import gc
import torch
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO
from Recommenders.BaseTempFolder import BaseTempFolder

from WWW2022.FastVAE_our_interface.FastVAE_our_interface import *
import torch.utils.data as dataloader

from WWW2022.FastVAE_our_interface.FastVAE_our_interface import SamplerBase, PopularSampler, MidxUniform, MidxUniPop
import torch
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from WWW2022.FastVAE_our_interface.FastVAE_our_interface import BaseVAE, VAE_Sampler
import argparse
import numpy as np
from WWW2022.FastVAE_our_interface.FastVAE_our_interface import Eval
import WWW2022.FastVAE_our_interface.FastVAE_our_interface
import logging
import datetime
import os
import time
import gc

# The cluster algorithmn(K-means) is implemented on the GPU
from operator import imod, neg
from numpy.core.numeric import indices
import scipy.sparse as sps
from sklearn import cluster
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn as nn
from torch._C import device, dtype


import numpy as np
import tensorflow as tf
import os
import shutil
import scipy.sparse as sps
from torch.utils.data import IterableDataset, Dataset

# from Conferences.CIKM.ExampleAlgorithm_github.main import get_model


class FastVAE_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "FastVAE_RecommenderWrapper"

    def __init__(self, config, URM_train, verbose=True, use_gpu=True):  # TODO
        super(FastVAE_RecommenderWrapper, self).__init__(
            URM_train, verbose=verbose)

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                print("GPU is not available, using cpu")
                self.device = torch.device("cpu:0")
        else:
            print("GPU is not available, using cpu")
            self.device = torch.device("cpu:0")

        print('device: ', self.device)
        config['device'] = self.device

        self.config = config

        sampler = str(self.config['sampler']) + '_' + \
            str(self.config['multi']) + 'x'
        if self.config['fix_seed']:
            setup_seed(self.config['seed'])

    def _compute_item_score(self, user_id_array, items_to_compute=None):  # TODO

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        print(user_id_array)

        allPred = t.mm(self.usrEmbeds[user_id_array], t.transpose(
            self.itmEmbeds, 1, 0)).detach().cpu().numpy()  # * (1 - trnMask) - trnMask * 1e8

        if items_to_compute is not None:
            item_scores[user_id_array,
                        items_to_compute] = allPred[user_id_array, items_to_compute]
        else:
            item_scores = allPred  # item_scores[user_id_array, :] = allPred

        if True:
            print(np.shape(item_scores))

        return item_scores

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        self._model = BaseVAE(  # TODO
            epochs=self.epochs,
        ).to(self.device)

    def fit(self,  # TODO
            epochs=None,
            temp_file_folder=None,
            # These are standard
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(
            input_temp_file_folder=temp_file_folder)

        self.epochs = epochs

        # Inizializza il modello
        self._init_model()
        # Ottimizzatore
        self.opt = t.optim.Adam(self._model.parameters(),  # TODO# TODO# TODO
                                lr=self.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(
            self.device, self.n_users, self.maskDepth, self.n_items, keepRate)
        self.sampler = LocalGraph(self.device, self.seedNum)

        ###############################################################################
        # This is a standard training with early stopping part, most likely you won't need to change it

        gc.collect()
        torch.cuda.empty_cache()

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        # serve per fare testing della compute item score
        # self.load_model("./result_experiments/__Temp_AutoCF_RecommenderWrapper_96339/", "_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

        self._prepare_model_for_validation()

    def _prepare_model_for_validation(self):  # TODOvvvvvvv

        self.usrEmbeds, self.itmEmbeds = self._model(
            self.torchBiAdj, self.torchBiAdj)

        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):  # TODO

        loss_, kld_loss = 0.0, 0.0
        if currentEpoch > 0:
            del sampler
            if self.config['sampler'] > 2:
                del item_emb

        infer_total_time = 0.0
        sample_total_time = 0.0
        loss_total_time = 0.0
        t0 = time.time()
        if self.config['sampler'] > 0:
            if self.config['sampler'] == 1:
                sampler = SamplerBase(
                    train_mat.shape[1] * self.config['multi'], self.config['sample_num'], device)
            elif self.config['sampler'] == 2:
                pop_count = np.squeeze(train_mat.sum(axis=0).A)
                pop_count = np.r_[pop_count, np.ones(
                    train_mat.shape[1] * (self.config['multi'] - 1))]
                sampler = PopularSampler(
                    pop_count, self.config['sample_num'], device)
            elif self.config['sampler'] == 3:
                item_emb = model._get_item_emb().detach()
                sampler = MidxUniform(
                    item_emb, self.config['sample_num'], device, self.config['cluster_num'])
            elif self.config['sampler'] == 4:
                item_emb = model._get_item_emb().detach()
                pop_count = np.squeeze(train_mat.sum(axis=0).A)
                pop_count = np.r_[pop_count, np.ones(
                    train_mat.shape[1] * (self.config['multi'] - 1))]
                sampler = MidxUniPop(
                    item_emb, self.config['sample_num'], device, self.config['cluster_num'], pop_count)
        t1 = time.time()

        for batch_idx, data in enumerate(train_dataloader):
            model.train()
            if self.config['sampler'] > 0:
                sampler.train()
            else:
                sampler = None
            pos_id = data
            pos_id = pos_id.to(device)

            optimizer.zero_grad()
            tt0 = time.time()
            mu, logvar, loss, sample_time, loss_time = _model(pos_id, sampler)
            tt1 = time.time()
            sample_total_time += sample_time
            infer_total_time += tt1 - tt0
            loss_total_time += loss_time

            kl_divergence = _model.kl_loss(
                mu, logvar, self.config['anneal'], reduction=self.config['reduction'])/self.config['batch_size']

            loss_ += loss.item()
            kld_loss += kl_divergence.item()
            loss += kl_divergence.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # break
            # torch.cuda.empty_cache()
            t2 = time.time()
        logger.info('--loss : %.2f, kl_dis : %.2f, total : %.2f ' %
                    (loss_, kld_loss, loss_ + kld_loss))
        torch.cuda.empty_cache()
        scheduler.step()
        gc.collect()
        initial_list.append(t1 - t0)
        training_list.append(t2 - t1)
        sampling_list.append(sample_total_time)
        inference_list.append(infer_total_time)
        cal_loss_list.append(loss_total_time)

        return

    def save_model(self, folder_path, file_name=None):  # TODO

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        data_dict_to_save = {
            'model': self._model,
        }

        t.save(data_dict_to_save, folder_path + file_name + '.mod')
        log('Model Saved: %s' % folder_path + file_name)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):  # TODO

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        ckp = t.load(folder_path + file_name + '.mod')
        self._model = ckp['model']
        self.opt = t.optim.Adam(self._model.parameters(),
                                lr=self.lr, weight_decay=0)

        self._print("Loading complete")


class UserItemData(Dataset):
    def __init__(self, train_mat, train_flag=True):
        super(UserItemData, self).__init__()
        self.train = train_mat
        if train_flag is True:
            self.users = np.random.permutation(self.train.shape[0])
        else:
            self.users = np.arange(self.train.shape[0])

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        # return self.user[idx], self.item[idx]
        pos_idx = self.train[self.users[idx]].nonzero()[1]
        return pos_idx
