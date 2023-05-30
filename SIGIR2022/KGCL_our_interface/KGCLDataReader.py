#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/02/2023

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Recommenders.DataIO import DataIO
import scipy.sparse as sps
import os, shutil, zipfile
import pandas as pd
import numpy as np


def _get_ICM_from_df(ICM_df, n_items):
    # The entities in the graph are the items + further entities-features
    # For the ICM we only keep the relations between items and entities (both direct and inverse relations)

    n_entities = max(ICM_df["head"].max(), ICM_df["tail"].max()) + 1
    ICM_df = ICM_df[ICM_df["head"]<n_items]
    ICM_sparse = sps.csr_matrix((np.ones(len(ICM_df), dtype=np.int),(ICM_df["head"].values, ICM_df["tail"].values)),
                          shape=(n_items, n_entities))

    ICM_sparse.data = np.ones_like(ICM_sparse.data)
    return ICM_sparse


def _reshape_sparse(URM, new_shape):

    if URM.shape[0] > new_shape[0] or URM.shape[1] > new_shape[1]:
        ValueError("new_shape cannot be smaller than SparseURMMatrix. URM shape is: {}, newShape is {}".format(
            URM.shape, new_shape))

    URM = sps.coo_matrix(URM)
    return sps.csr_matrix((URM.data, (URM.row, URM.col)), shape=new_shape)


def _resize_to_maximum_shape(URM_list):

    n_row, n_col = URM_list[0].shape

    for URM in URM_list[1:]:
        n_row = max(n_row, URM.shape[0])
        n_col = max(n_col, URM.shape[1])

    for index in range(len(URM_list)):
        URM_list[index] = _reshape_sparse(URM_list[index], (n_row, n_col))

    return URM_list


class KGCLDataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, split_type, pre_splitted_path, freeze_split = True):
        super(KGCLDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

        original_data_path = os.path.join("SIGIR2022/KGCL_github/data", dataset_name)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print("{}: Attempting to load saved data from {}".format(dataset_name, pre_splitted_path + pre_splitted_filename))
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            if freeze_split:
                raise Exception("Splitted data not found!")

            print("{}: Pre-splitted data not found, building new one".format(dataset_name))
            print("{}: loading data".format(dataset_name))

            URM_train_validation = self._load_data_file(os.path.join(original_data_path, 'train.txt'))
            URM_test = self._load_data_file(os.path.join(original_data_path, 'test.txt'))
            URM_train_validation, URM_test = _resize_to_maximum_shape([URM_train_validation, URM_test])

            self.knowledge_base_df = pd.read_csv(os.path.join(original_data_path, 'kg.txt'), header=None, sep=" ").drop_duplicates()

            self.knowledge_base_df.columns = ["head", "relation", "tail"]
            self.knowledge_base_df.drop_duplicates()

            # Split is 72% training, 8% validation and 20% test
            if split_type == "ours":
                URM_all = URM_train_validation + URM_test
                URM_train_validation, URM_test = split_train_in_two_percentage_user_wise(URM_all, train_percentage=0.80)

            # Split user-wise
            URM_train, URM_validation = split_train_in_two_percentage_user_wise(URM_train_validation, train_percentage=0.90)

            _, n_items = URM_train.shape

            self.ICM_DICT = {
                "ICM_entities": _get_ICM_from_df(self.knowledge_base_df.copy(), n_items)
            }
            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
                "knowledge_base_df": self.knowledge_base_df,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("{}: loading complete".format(dataset_name))


    def _load_data_file(self, filePath, separator = " "):

        URM_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, auto_create_col_mapper=False)

        fileHandle = open(filePath, "r", encoding = "utf-8")

        for line in fileHandle:
            if (len(line)) > 1:
                line = line.replace("\n", "")
                line = line.split(separator)

                # Avoid parsing users with no interactions
                if len(line[1]) > 0:
                    line = [int(line[i]) for i in range(len(line))]
                    URM_builder.add_single_row(line[0], line[1:], data=1.0)

        fileHandle.close()

        return  URM_builder.get_SparseMatrix()

class KGCL_Add_KB_to_existing_split_DataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, collaborative_dataset, pre_splitted_path, freeze_split=True):
        super(KGCL_Add_KB_to_existing_split_DataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

        original_data_path = os.path.join("SIGIR2022/KGCL_github/data", dataset_name)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print("{}: Attempting to load saved data from {}".format(dataset_name,
                                                                     pre_splitted_path + pre_splitted_filename))
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:

            if freeze_split:
                raise Exception("Splitted data not found!")

            print("{}: Pre-splitted data not found, building new one".format(dataset_name))
            print("{}: loading data".format(dataset_name))

            self.knowledge_base_df = pd.read_csv(os.path.join(original_data_path, 'kg.txt'), header=None, sep=" ").drop_duplicates()

            self.knowledge_base_df.columns = ["head", "relation", "tail"]
            self.knowledge_base_df.drop_duplicates()

            _, n_items = collaborative_dataset.URM_DICT["URM_train"].shape

            self.ICM_DICT = {
                "ICM_entities": _get_ICM_from_df(self.knowledge_base_df.copy(), n_items)
            }

            self.UCM_DICT = collaborative_dataset.UCM_DICT
            self.URM_DICT = collaborative_dataset.URM_DICT

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
                "knowledge_base_df": self.knowledge_base_df,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("{}: loading complete".format(dataset_name))
