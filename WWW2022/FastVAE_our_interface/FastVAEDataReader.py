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
from scipy import sparse
import random as rd
import numpy as np
import math
import os
from Data_manager.DataReader_utils import download_from_URL, remove_Dataframe_duplicates
from Data_manager.DataReader_utils import remove_empty_rows_and_cols
from pandas.api.types import CategoricalDtype

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
            __file__), "../FastVAE_github/datasets")
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

            if dataset_name == "ml10M":
                dataset_file = dataset_dir + 'data.mat'

                mat = self.load_file(filename=dataset_file)

            elif dataset_name == "gowalla":
                dataset_file = dataset_dir + '.txt'
                print("Loading Interactions")
                URM_all_dataframe = pd.read_csv(filepath_or_buffer=dataset_file, sep="\t", header=None,
                                                dtype={0: str, 4: str}, usecols=[0, 4])

                URM_all_dataframe.columns = ["UserID", "ItemID"]
                URM_all_dataframe["Data"] = 1

                URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe,
                                                                unique_values_in_columns=[
                                                                    'UserID', 'ItemID'],
                                                                keep_highest_value_in_col="Data")

                users = URM_all_dataframe["UserID"].unique()
                items = URM_all_dataframe["ItemID"].unique()
                shape = (len(users), len(items))

                # Create indices for users and movies
                user_cat = CategoricalDtype(
                    categories=sorted(users), ordered=True)
                item_cat = CategoricalDtype(
                    categories=sorted(items), ordered=True)
                user_index = URM_all_dataframe["UserID"].astype(
                    user_cat).cat.codes
                item_index = URM_all_dataframe["ItemID"].astype(
                    item_cat).cat.codes

                # Conversion via COO matrix
                coo = sparse.coo_matrix(
                    (URM_all_dataframe["Data"], (user_index, item_index)), shape=shape)
                mat = coo.tocsr()

            print(mat.get_shape())
            # Apply required min user interactions ORIGINAL split

            mat_modified, users_to_remove, items_to_remove = select_users_with_min_interactions(
                mat, min_interactions=10)
            print(mat_modified.get_shape())

            mat_transposed = mat_modified.transpose(copy=True)

            mat_final_transposed, users_to_remove, items_to_remove = select_users_with_min_interactions(
                mat_transposed, min_interactions=10)
            mat_final = mat_final_transposed.transpose()
            print(mat_final.get_shape())

            while (mat_modified.get_shape() != mat_final.get_shape()):
                mat_modified, users_to_remove, items_to_remove = select_users_with_min_interactions(
                    mat_final, min_interactions=10)
                print(mat_modified.get_shape())

                mat_transposed = mat_modified.transpose(copy=True)

                mat_final_transposed, users_to_remove, items_to_remove = select_users_with_min_interactions(
                    mat_transposed, min_interactions=10)
                mat_final = mat_final_transposed.transpose()
                print(mat_final.get_shape())

            mat = mat_final

            URM_train_temp, URM_test = self.split_matrix(mat)
            URM_train, URM_validation = self.split_matrix(URM_train_temp)

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


def select_users_with_min_interactions(URM, min_interactions=10, reshape=True):
    """

    :param URM:
    :param min_interactions:
    :param reshape:
    :return: URM, removedUsers, removedItems
    """

    print("DataReaderPostprocessing_User_min_interactions: min_interactions extraction will zero out some users and items without changing URM shape")

    URM.eliminate_zeros()

    n_users = URM.shape[0]
    n_items = URM.shape[1]

    print("DataReaderPostprocessing_User_min_interactions: Initial URM desity is {:.2E}".format(
        URM.nnz/(n_users*n_items)))

    n_users, n_items = URM.shape

    URM = sps.csr_matrix(URM)
    user_to_remove_mask = np.ediff1d(URM.indptr) < min_interactions
    removed_users = np.arange(0, n_users, dtype=np.int)[user_to_remove_mask]

    for user in removed_users:
        start_pos = URM.indptr[user]
        end_pos = URM.indptr[user + 1]

        URM.data[start_pos:end_pos] = np.zeros_like(
            URM.data[start_pos:end_pos])

    URM.eliminate_zeros()

    URM = sps.csc_matrix(URM)
    items_to_remove_mask = np.ediff1d(URM.indptr) == 0
    removed_items = np.arange(0, n_items, dtype=np.int)[items_to_remove_mask]

    if URM.data.sum() == 0:
        print("DataReaderPostprocessing_User_min_interactions: WARNING URM is empty.")

    else:
        print("DataReaderPostprocessing_User_min_interactions: URM desity without zeroed-out nodes is {:.2E}.\n"
              "Users with less than {} interactions are {} ({:4.1f}%), Items are {} ({:4.1f}%)".format(
                  sum(URM.data)/((n_users-len(removed_users))
                                 * (n_items-len(removed_items))),
                  min_interactions,
                  len(removed_users), len(removed_users)/n_users*100,
                  len(removed_items), len(removed_items)/n_items*100))

    print("DataReaderPostprocessing_User_min_interactions: split complete")

    URM = sps.csr_matrix(URM)

    if reshape:
        # Remove all columns and rows with no interactions
        return remove_empty_rows_and_cols(URM)

    return URM.copy(), removed_users, removed_items
