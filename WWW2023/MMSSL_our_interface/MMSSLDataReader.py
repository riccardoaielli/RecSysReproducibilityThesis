#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/06/2023

@author: Anonymized for blind review
"""

# Classe che prende i dati e li trasforma in matrici sparse


import pandas as pd

from tqdm import tqdm
import os
import numpy as np
import pickle
import json
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.data.graph_serialize import *

# from dgl.nn import EdgeWeightNorm

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# from Data_manager.split_functions.split_train_validation_random_holdout import \
#    split_train_in_two_percentage_global_sample

import os
import numpy as np
import pandas as pd
import scipy.sparse as sps

from Recommenders.DataIO import DataIO

from scipy import *
import scipy.linalg as la
from scipy.sparse import *

from WWW2023.MMSSL_our_interface.MMSSL.utility.load_data import Data


class MMSSLDataReader(object):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split=False):
        super(MMSSLDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        print(pre_splitted_path)
        pre_splitted_filename = "splitted_data"

        base_ds_path = os.path.join(os.path.dirname(
            __file__), "../MMSSL_github/data")
        dataset_dir = os.path.join(base_ds_path, dataset_name)
        print(dataset_dir)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print(self.__class__.__name__ + ": Attempting to load saved data from " +
                  pre_splitted_path + pre_splitted_filename)
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            if freeze_split:
                raise Exception("Splitted data not found!")

            print(self.__class__.__name__ +
                  ": Pre-splitted data not found, building new one")
            print(self.__class__.__name__ + ": loading data")

            self.image_feats = np.load(dataset_dir + '/image_feat.npy')
            self.text_feats = np.load(dataset_dir + '/text_feat.npy')
            self.image_feat_dim = self.image_feats.shape[-1]
            self.text_feat_dim = self.text_feats.shape[-1]
            self.ui_graph = self.ui_graph_raw = pickle.load(
                open(dataset_dir + '/train_mat', 'rb'))

            train_file = dataset_dir + '/train.json'
            val_file = dataset_dir + '/val.json'
            test_file = dataset_dir + '/test.json'

            # get number of users and items
            self.n_users, self.n_items = 0, 0
            self.n_train, self.n_test, self.n_val = 0, 0, 0
            self.neg_pools = {}

            self.exist_users = []

            train = json.load(open(train_file))
            test = json.load(open(test_file))
            val = json.load(open(val_file))
            for uid, items in train.items():
                if len(items) == 0:
                    continue
                uid = int(uid)
                self.exist_users.append(uid)
                self.n_items = max(self.n_items, max(items))
                self.n_users = max(self.n_users, uid)
                self.n_train += len(items)

            for uid, items in test.items():
                uid = int(uid)
                try:
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
                except:
                    continue

            for uid, items in val.items():
                uid = int(uid)
                try:
                    self.n_items = max(self.n_items, max(items))
                    self.n_val += len(items)
                except:
                    continue

            self.n_items += 1
            self.n_users += 1

            print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
            print('n_interactions=%d' %
                  (self.n_train + self.n_test + self.n_val))
            print('n_train=%d, n_test=%d, n_val=%d, sparsity=%.5f' % (self.n_train, self.n_test,
                  self.n_val, (self.n_train + self.n_test + self.n_val)/(self.n_users * self.n_items)))

            self.train = sp.dok_matrix(
                (self.n_users, self.n_items), dtype=np.float32)
            self.test = sp.dok_matrix(
                (self.n_users, self.n_items), dtype=np.float32)
            self.val = sp.dok_matrix(
                (self.n_users, self.n_items), dtype=np.float32)

            self.R_Item_Interacts = sp.dok_matrix(
                (self.n_items, self.n_items), dtype=np.float32)

            self.train_items, self.test_set, self.val_set = {}, {}, {}
            for uid, train_items in train.items():
                if len(train_items) == 0:
                    continue
                uid = int(uid)
                for idx, i in enumerate(train_items):
                    self.train[uid, i] = 1.

                self.train_items[uid] = train_items

            for uid, test_items in test.items():
                if len(test_items) == 0:
                    continue
                uid = int(uid)
                for idx, i in enumerate(test_items):
                    self.test[uid, i] = 1.

                self.test_set[uid] = test_items

            for uid, val_items in val.items():
                if len(val_items) == 0:
                    continue
                uid = int(uid)
                for idx, i in enumerate(val_items):
                    self.val[uid, i] = 1.

                self.val_set[uid] = val_items

            # data_generator = Data(path=dataset_dir, batch_size=batch_size)
            # print(self.train.shape)
            # print(self.train.todense())
            # print(self.test.shape)
            # print(self.test.todense())
            # print(self.val.shape)
            # print(self.val.todense())

            self.URM_DICT = {
                "URM_train": self.train.tocsr(),
                "URM_test": self.test.tocsr(),
                "URM_validation": self.val.tocsr(),
            }

            self.ICM_DICT = {}

            self.UCM_DICT = {}

            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
                "image_feats": self.image_feats,
                "text_feats": self.text_feats,
                "image_feat_dim": self.image_feat_dim,
                "text_feat_dim": self.text_feat_dim,
                "ui_graph": self.ui_graph,
                "n_items": self.n_items,
                "n_users": self.n_users,
                "n_train": self.n_train,
                "test_set": self.test_set,
                "val_set": self.val_set,
                "exist_users": self.exist_users,
                "train_items": self.train_items,
            }

            print('saving data in splitted_data.zip')
            dataIO.save_data(pre_splitted_filename,
                             data_dict_to_save=data_dict_to_save)

            print(self.__class__.__name__ + ": loading complete")
