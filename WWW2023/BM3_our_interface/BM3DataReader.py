#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/06/2023

@author: Anonymized for blind review
"""

# Classe che prende i dati e li trasforma in matrici sparse


import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.data.graph_serialize import *
from WWW2023.BM3_our_interface.utils.utils import init_seed, get_model, get_trainer, dict2str
from logging import getLogger
from itertools import product
from WWW2023.BM3_our_interface.utils.dataset import RecDataset
from WWW2023.BM3_our_interface.utils.dataloader import TrainDataLoader, EvalDataLoader
from WWW2023.BM3_our_interface.utils.logger import init_logger
from WWW2023.BM3_our_interface.utils.configurator import Config
from WWW2023.BM3_our_interface.utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import math
import torch
import random
import numpy as np
from scipy.sparse import coo_matrix
# from dgl.nn import EdgeWeightNorm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
# from Data_manager.split_functions.split_train_validation_random_holdout import \
#    split_train_in_two_percentage_global_sample
import scipy.sparse as sps
from Recommenders.DataIO import DataIO
from scipy import *
import scipy.linalg as la
from scipy.sparse import *


class BM3DataReader(object):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, device, freeze_split=False):
        super(BM3DataReader, self).__init__()

        base_ds_path = os.path.join(os.path.dirname(
            __file__), "../BM3_github/data")
        dataset_dir = os.path.join(base_ds_path, dataset_name)
        print(dataset_dir)

        print(self.__class__.__name__ +
              ": Pre-splitted data not found, building new one")
        print(self.__class__.__name__ + ": loading data")

        # merge config dict
        self.config = Config(model='BM3', dataset=dataset_name)
        self.config.final_config_dict['device'] = device
        init_logger(self.config)
        logger = getLogger()
        # print config infor
        logger.info('██Server: \t' + platform.node())
        logger.info('██Dir: \t' + os.getcwd() + '\n')
        logger.info(self.config)

        self.config.final_config_dict['data_path'] = base_ds_path

        # load data
        dataset = RecDataset(self.config)
        # print dataset statistics
        logger.info(str(dataset))

        train_dataset, valid_dataset, test_dataset = dataset.split()
        logger.info('\n====Training====\n' + str(train_dataset))
        logger.info('\n====Validation====\n' + str(valid_dataset))
        logger.info('\n====Testing====\n' + str(test_dataset))

        # wrap into dataloader
        self.train_data = TrainDataLoader(
            self.config, train_dataset, batch_size=self.config['train_batch_size'], shuffle=False)  # TODO originariamente shuffle=True
        self.valid_data = EvalDataLoader(
            self.config, valid_dataset, additional_dataset=train_dataset, batch_size=self.config['eval_batch_size'])
        self.test_data = EvalDataLoader(
            self.config, test_dataset, additional_dataset=train_dataset, batch_size=self.config['eval_batch_size'])

        URM_train = self.train_data.inter_matrix(form='csr')
        URM_test = self.valid_data.inter_matrix(form='csr')
        URM_validation = self.test_data.inter_matrix(form='csr')

        self.URM_DICT = {
            "URM_train": URM_train,
            "URM_test": URM_test,
            "URM_validation": URM_validation,
        }

        # print(URM_train.shape)
        # print(URM_train)
        # print(URM_test.shape)
        # print(URM_test)
        # print(URM_validation.shape)
        # print(URM_validation)

        self.ICM_DICT = {}

        self.UCM_DICT = {}

        print(self.__class__.__name__ + ": loading complete")
