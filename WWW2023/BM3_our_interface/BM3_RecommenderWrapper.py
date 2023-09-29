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

from WWW2023.BM3_our_interface.models.bm3 import *
from WWW2023.BM3_our_interface.utils.utils import init_seed, get_model, get_trainer, dict2str
import torch.utils.data as dataloader
import torch.optim as optim

import numpy as np
import tensorflow as tf
import os
import shutil
import scipy.sparse as sps

# from Conferences.CIKM.ExampleAlgorithm_github.main import get_model


class BM3_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "BM3_RecommenderWrapper"

    def __init__(self, URM_train, config, train_data, test_data, valid_data, verbose=True, use_gpu=True):
        super(BM3_RecommenderWrapper, self).__init__(
            URM_train, verbose=verbose)

        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.batcvalid_datah = valid_data

        # Dataset loadded, run model
        hyper_ret = []
        val_metric = config['valid_metric'].lower()
        best_test_value = 0.0
        idx = best_test_idx = 0

        # hyper-parameters
        hyper_ls = []
        if "seed" not in config['hyper_parameters']:
            config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
        for i in config['hyper_parameters']:
            hyper_ls.append(config[i] or [None])

        # TODO Fa il tuning degli iperparametri o meglio penso
        # combinations
        combinators = list(product(*hyper_ls))
        total_loops = len(combinators)
        for hyper_tuple in combinators:
            # random seed reset
            for j, k in zip(config['hyper_parameters'], hyper_tuple):
                config[j] = k
            init_seed(config['seed'])

            # TODO trainer loading and initialization, cos'è questo trainer?
            trainer = get_trainer()(config, model)
            # debug
            # model training
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(
                train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
            #########
            hyper_ret.append(
                (hyper_tuple, best_valid_result, best_test_upon_valid))

            # save best test
            if best_test_upon_valid[val_metric] > best_test_value:
                best_test_value = best_test_upon_valid[val_metric]
                best_test_idx = idx
            idx += 1

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

        # set random state of dataloader
        self.train_data.pretrain_setup()

        self._model = BM3(self.config,
                          self.train_data,
                          ).to(self.config['device'])

    def fit(self,  # TODO
            epochs=None,
            lr=None,
            latdim=None,
            ssl_reg=None,
            decay=None,
            head=None,
            gcn_layer=None,
            gt_layer=None,
            tstEpoch=None,
            seedNum=None,
            reg=None,
            maskDepth=None,
            fixSteps=None,
            keepRate=None,
            eps=None,

            temp_file_folder=None,
            # These are standard
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(
            input_temp_file_folder=temp_file_folder)

        self.lr = lr  # TODO
        self.epochs = epochs
        self.latdim = latdim
        self.reg = reg
        self.ssl_reg = ssl_reg
        self.decay = decay
        self.head = head
        self.gcn_layer = gcn_layer
        self.gt_layer = gt_layer
        self.tstEpoch = tstEpoch
        self.seedNum = seedNum
        self.maskDepth = maskDepth
        self.fixSteps = fixSteps
        self.keepRate = keepRate
        self.eps = eps

        # Inizializza il modello
        self._init_model()
        # Ottimizzatore
        self.learner = self.config['learner']
        if self.learner.lower() == 'adam':
            self.opt = optim.Adam(
                self._model.parameters(), lr=self.lr)
        else:
            self.logger.warning(
                'Received unrecognized optimizer, set default Adam optimizer')

        # TODO scheduler

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

        self._model.train()
        self.optimizer.zero_grad()
        self.optimizer.step()
        self.lr_scheduler.step()

        return

    def save_model(self, folder_path, file_name=None):  # TODO
        """ if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        data_dict_to_save = {
            'model': self._model,
        }

        t.save(data_dict_to_save, folder_path + file_name + '.mod')
        log('Model Saved: %s' % folder_path + file_name)

        self._print("Saving complete") """

    def load_model(self, folder_path, file_name=None):  # TODO
        """ if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        ckp = t.load(folder_path + file_name + '.mod')
        self._model = ckp['model']
        self.opt = t.optim.Adam(self._model.parameters(),
                                lr=self.lr, weight_decay=0)

        self._print("Loading complete") """
