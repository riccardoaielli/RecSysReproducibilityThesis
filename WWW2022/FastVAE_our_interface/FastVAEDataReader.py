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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pandas as pd
from torch.utils.data import IterableDataset, Dataset
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import scipy.io as sci
import scipy as sp
import random as rd
import numpy as np
import math
import os

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


class FastVAEDataReader(object):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split=False):
        super(FastVAEDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        print(pre_splitted_path)
        pre_splitted_filename = "splitted_data"

        base_ds_path = os.path.join(os.path.dirname(
            __file__), "../FastVAE_github/datasets")  # TODO
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

            # TODO Fai la load da file
            # TODO setta tutti i valori nll'oggetto
            # self.graph = graph

        except FileNotFoundError:

            if freeze_split:
                raise Exception("Splitted data not found!")

            print(self.__class__.__name__ +
                  ": Pre-splitted data not found, building new one")
            print(self.__class__.__name__ + ": loading data")

            mat = self.load_file(filename=dataset_dir + 'data.mat')
            URM_train_temp, URM_test = self.split_matrix(mat)
            URM_train, URM_validation = self.split_matrix(URM_train_temp)

            # TODO

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            self.ICM_DICT = {}

            self.UCM_DICT = {}

            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            print('saving data in splitted_data.zip')
            dataIO.save_data(pre_splitted_filename,
                             data_dict_to_save=data_dict_to_save)

            print(self.__class__.__name__ + ": loading complete")

    def load_file(self, filename=''):
        # if file_name.endswith('.mat'):
        #     return sci.loadmat(file_name)['data']
        # else:
        #     raise ValueError('not supported file type')
        if filename.endswith('.mat'):
            return sci.loadmat(filename)['data']
        elif filename.endswith('.txt') or filename.endswith('.tsv'):
            sep = '\t'
        elif filename.endswith('.csv'):
            sep = ','
        else:
            raise ValueError('not supported file type')
        max_user = -1
        max_item = -1
        row_idx = []
        col_idx = []
        data = []
        for line in open(filename):
            user, item, rating = line.strip().split(sep)
            user, item, rating = int(user) - 1, int(item)-1, float(rating)
            row_idx.append(user)
            col_idx.append(item)
            data.append(rating)
            if user > max_user:
                max_user = user
            if item > max_item:
                max_item = item
        return sp.sparse.csc_matrix((data, (row_idx, col_idx)), (max_user+1, max_item+1))

    def split_matrix(self, mat, ratio=0.8):
        mat = mat.tocsr()  # 按行读取，即每一行为一个用户
        m, n = mat.shape
        train_data_indices = []
        train_indptr = [0] * (m+1)
        test_data_indices = []
        test_indptr = [0] * (m+1)
        for i in range(m):
            row = [(mat.indices[j], mat.data[j])
                   for j in range(mat.indptr[i], mat.indptr[i+1])]
            train_idx = rd.sample(range(len(row)), round(ratio * len(row)))
            train_binary_idx = np.full(len(row), False)
            train_binary_idx[train_idx] = True
            test_idx = (~train_binary_idx).nonzero()[0]
            for idx in train_idx:
                train_data_indices.append(row[idx])
            train_indptr[i+1] = len(train_data_indices)
            for idx in test_idx:
                test_data_indices.append(row[idx])
            test_indptr[i+1] = len(test_data_indices)

        [train_indices, train_data] = zip(*train_data_indices)
        [test_indices, test_data] = zip(*test_data_indices)

        train_mat = sp.sparse.csr_matrix(
            (train_data, train_indices, train_indptr), (m, n))
        test_mat = sp.sparse.csr_matrix(
            (test_data, test_indices, test_indptr), (m, n))
        return train_mat, test_mat
