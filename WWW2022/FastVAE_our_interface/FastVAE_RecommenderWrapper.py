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

    def __init__(self, URM_train, config, verbose=True, use_gpu=True):  # TODO
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
        self.train_mat = URM_train

        self.sampler = str(self.config['sampler'][0]) + '_' + \
            str(self.config['multi'][0]) + 'x'

        self.item_emb = None
        if self.config['fix_seed'][0]:
            setup_seed(int(self.config['seed'][0]))

    def _compute_item_score(self, user_id_array, items_to_compute=None):  # TODO

        item_scores = - np.ones((len(user_id_array), self.item_num)) * np.inf

        allPred = torch.mm(self.user_emb[user_id_array], torch.transpose(
            self.item_emb, 1, 0)).detach().cpu().numpy()

        # users = np.random.choice(user_num, min(user_num, 5000), False)
        # m = Eval.evaluate_item(
        #     self.train_mat[users, :], self.test_mat[users, :], user_emb[users, :], item_emb, topk=0)

        item_scores = allPred

        # print(user_id_array)

        # allPred = t.mm(self.usrEmbeds[user_id_array], t.transpose(
        #     self.itmEmbeds, 1, 0)).detach().cpu().numpy()  # * (1 - trnMask) - trnMask * 1e8

        # if items_to_compute is not None:
        #     item_scores[user_id_array,
        #                 items_to_compute] = allPred[user_id_array, items_to_compute]
        # else:
        #     item_scores = allPred  # item_scores[user_id_array, :] = allPred

        # if True:
        #     print(np.shape(item_scores))

        return item_scores

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        if self.config['sampler'][0] == 0:
            self._model = BaseVAE(
                self.item_num * self.config['multi'][0], self.config['dim'][0]).to(self.device)
        elif self.config['sampler'][0] > 0:
            self._model = VAE_Sampler(
                self.item_num * self.config['multi'][0], self.config['dim'][0]).to(self.device)
        else:
            raise ValueError('Not supported model name!!!')

        # self._model = BaseVAE(  # TODO
        #     epochs=self.epochs,
        # ).to(self.device)

    def fit(self,  # TODO
            epochs=None,
            temp_file_folder=None,
            # These are standard
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(
            input_temp_file_folder=temp_file_folder)

        self.user_num, self.item_num = self.train_mat.shape

        assert self.config['sample_num'][0] < self.item_num

        # Inizializza il modello
        self._init_model()
        # Ottimizzatore
        if self.config['optim'][0] == 'adam':
            self.optimizer = torch.optim.Adam(self._model.parameters(
            ), lr=self.config['learning_rate'][0], weight_decay=self.config['weight_decay'][0])
        elif self.config['optim'][0] == 'sgd':
            self.optimizer = torch.optim.SGD(self._model.parameters(
            ), lr=self.config['learning_rate'][0], weight_decay=self.config['weight_decay'][0])
        else:
            raise ValueError('Unkown optimizer!')

        self.scheduler = StepLR(
            self.optimizer, self.config['step_size'][0], self.config['gamma'][0])

        train_data = UserItemData(self.URM_train)
        self.train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'][0], num_workers=self.config['num_workers'][0],
                                           pin_memory=True, shuffle=True, collate_fn=custom_collate_)

        self.initial_list = []
        self.training_list = []
        self.sampling_list = []
        self.inference_list = []
        self.cal_loss_list = []

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

    def _prepare_model_for_validation(self):

        self._model.eval()
        with torch.no_grad():

            user_emb = get_user_embs(
                self.train_mat, self._model, self.device, self.config)
            item_emb = self._model._get_item_emb()

            # user_emb = user_emb.cpu().data
            # item_emb = item_emb.cpu().data

            self.user_emb = user_emb.cpu().data
            self.item_emb = item_emb.cpu().data

        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):

        loss_, kld_loss = 0.0, 0.0
        if currentEpoch > 0:
            del self.sampler
            if self.config['sampler'][0] > 2:
                del self.item_emb

        infer_total_time = 0.0
        sample_total_time = 0.0
        loss_total_time = 0.0
        t0 = time.time()
        if self.config['sampler'][0] > 0:
            if self.config['sampler'][0] == 1:
                self.sampler = SamplerBase(
                    self.train_mat.shape[1] * self.config['multi'][0], self.config['sample_num'][0], self.device)
            elif self.config['sampler'][0] == 2:
                pop_count = np.squeeze(self.train_mat.sum(axis=0).A)
                pop_count = np.r_[pop_count, np.ones(
                    self.train_mat.shape[1] * (self.config['multi'][0] - 1))]
                self.sampler = PopularSampler(
                    pop_count, self.config['sample_num'][0], self.device)
            elif self.config['sampler'][0] == 3:
                self.item_emb = self._model._get_item_emb().detach()
                self.sampler = MidxUniform(
                    self.item_emb, self.config['sample_num'][0], self.device, self.config['cluster_num'][0])
            elif self.config['sampler'][0] == 4:
                self.item_emb = self._model._get_item_emb().detach()
                pop_count = np.squeeze(self.train_mat.sum(axis=0).A)
                pop_count = np.r_[pop_count, np.ones(
                    self.train_mat.shape[1] * (self.config['multi'][0] - 1))]
                self.sampler = MidxUniPop(
                    self.item_emb, self.config['sample_num'][0], self.device, self.config['cluster_num'][0], pop_count)
        t1 = time.time()

        for batch_idx, data in enumerate(self.train_dataloader):
            self._model.train()
            if self.config['sampler'][0] > 0:
                self.sampler.train()
            else:
                self.sampler = None
            pos_id = data
            pos_id = pos_id.to(self.device)

            self.optimizer.zero_grad()
            tt0 = time.time()
            mu, logvar, loss, sample_time, loss_time = self._model(
                pos_id, self.sampler)
            tt1 = time.time()
            sample_total_time += sample_time
            infer_total_time += tt1 - tt0
            loss_total_time += loss_time

            kl_divergence = self._model.kl_loss(
                mu, logvar, self.config['anneal'][0], reduction=self.config['reduction'][0])/self.config['batch_size'][0]

            loss_ += loss.item()
            kld_loss += kl_divergence.item()
            loss += kl_divergence.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            self.optimizer.step()
            # break
            # torch.cuda.empty_cache()
            t2 = time.time()

        torch.cuda.empty_cache()
        self.scheduler.step()
        gc.collect()
        self.initial_list.append(t1 - t0)
        self.training_list.append(t2 - t1)
        self.sampling_list.append(sample_total_time)
        self.inference_list.append(infer_total_time)
        self.cal_loss_list.append(loss_total_time)

        return

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        data_dict_to_save = {
            'model': self._model,
        }

        torch.save(data_dict_to_save, folder_path + file_name + '.mod')
        self._print('Model Saved: %s' % folder_path + file_name)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        ckp = torch.load(folder_path + file_name + '.mod')
        self._model = ckp['model']
        if self.config['optim'][0] == 'adam':
            self.optimizer = torch.optim.Adam(self._model.parameters(
            ), lr=self.config['learning_rate'][0], weight_decay=self.config['weight_decay'][0])
        elif self.config['optim'][0] == 'sgd':
            self.optimizer = torch.optim.SGD(self._model.parameters(
            ), lr=self.config['learning_rate'][0], weight_decay=self.config['weight_decay'][0])
        else:
            raise ValueError('Unkown optimizer!')
        self.scheduler = StepLR(
            self.optimizer, self.config['step_size'][0], self.config['gamma'][0])

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


def custom_collate_(batch):
    # TODO numpy.array()
    return torch.LongTensor(np.array(pad_sequence_int(batch)))


def get_user_embs(data_mat, model, device, config):
    data = UserItemData(data_mat, train_flag=False)
    dataloader = DataLoader(data, batch_size=config['batch_size_u'][0], num_workers=config['num_workers'][0],
                            pin_memory=False, shuffle=False, collate_fn=custom_collate_)
    user_lst = []
    for e in dataloader:
        user_his = e
        user_emb = model._get_user_emb(user_his.to(device))
        user_lst.append(user_emb)
    return torch.cat(user_lst, dim=0)
