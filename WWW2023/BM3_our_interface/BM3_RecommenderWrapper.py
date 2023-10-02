#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""
import gc
import torch
import pandas as pd
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO
from Recommenders.BaseTempFolder import BaseTempFolder

from WWW2023.BM3_our_interface.models.bm3 import *
from WWW2023.BM3_our_interface.utils.utils import init_seed, get_model, get_trainer, dict2str
import torch.utils.data as dataloader
import torch.optim as optim
from itertools import product
from logging import getLogger
from WWW2023.BM3_our_interface.common import *

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

        logger = getLogger()

        # hyper-parameters
        """ hyper_ls = []
        if "seed" not in config['hyper_parameters']:
            config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
        for i in config['hyper_parameters']:
            hyper_ls.append(config[i] or [None]) """

        # TODO Prova tutte le combinazioni di iperparametri
        # combinations
        """ combinators = list(product(*hyper_ls))
        total_loops = len(combinators)
        for hyper_tuple in combinators:
            # random seed reset
            for j, k in zip(config['hyper_parameters'], hyper_tuple):
                config[j] = k
            init_seed(config['seed'])

            logger.info('========={}/{}: Parameters:{}={}======='.format(
                idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

            # Istanzia il trainer e appena dopo chiama la fit mi sembra u parallelo del wrapper e della init model
            trainer = get_trainer()(config, _model)
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

            logger.info('best valid result: {}'.format(
                dict2str(best_valid_result)))
            logger.info('test result: {}'.format(
                dict2str(best_test_upon_valid)))
            logger.info('████Current BEST████:\nParameters: {}={},\n'
                        'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                                                            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

        # log info
        logger.info('\n============All Over=====================')
        for (p, k, v) in hyper_ret:
            logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                      p, dict2str(k), dict2str(v)))

        logger.info('\n\n█████████████ BEST ████████████████')
        logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                            hyper_ret[best_test_idx][0],
                                                                            dict2str(
                                                                                hyper_ret[best_test_idx][1]),
                                                                            dict2str(hyper_ret[best_test_idx][2]))) """

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        self._model.eval()

        scores = self._model.full_sort_predict(
            user_id_array).detach().cpu().numpy()  # TODO

        if items_to_compute is not None:  # penso sia sbagliata
            item_scores[user_id_array,
                        items_to_compute] = scores[user_id_array, items_to_compute]
        else:
            item_scores = scores

        return item_scores

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        # set random state of dataloader
        # self.train_data.pretrain_setup() # TODO rimuovo perchè penso che sballi tutti gli score

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
            dropout=None,
            embedding_size=None,

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
        self.dropout = dropout
        self.embedding_size = self.latdim

        # Inizializza il modello
        self._init_model()
        # Ottimizzatore
        self.learner = self.config['learner']
        if self.learner.lower() == 'adam':
            self.optimizer = optim.Adam(
                self._model.parameters(), lr=self.lr)
            print("Adam Optimizer")
        else:
            self.logger.warning(
                'Received unrecognized optimizer, set default Adam optimizer')

        # TODO scheduler
        # fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = self.config['learning_rate_scheduler']
        def fac(epoch): return lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

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
        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):

        # train
        self._model.pre_epoch_processing()
        self._model.train()
        for batch_idx, interaction in enumerate(self.train_data):
            self.optimizer.zero_grad()
            self.optimizer.step()
        # if torch.is_tensor(train_loss):
        #     # get nan loss
        #     break
        self.lr_scheduler.step()
        self._model.post_epoch_processing()

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
        print('Model Saved: %s' % folder_path + file_name)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        ckp = torch.load(folder_path + file_name + '.mod')
        self._model = ckp['model']
        self.optimizer = optim.Adam(
            self._model.parameters(), lr=self.lr)

        self._print("Loading complete")
