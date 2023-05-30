#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/01/2023

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import os
from Recommenders.DataIO import DataIO
import scipy.sparse as sps
import numpy as np
import pandas as pd


def _get_ICM_from_df(knowledge_base_df, n_items):
    # The entities in the graph are the items + further entities-features
    # For the ICM we only keep the relations between items and entities (both direct and inverse relations)

    knowledge_base_direct = knowledge_base_df[(knowledge_base_df["head"] < n_items) & (knowledge_base_df["tail"] >= n_items)]
    knowledge_base_inverse = knowledge_base_df[(knowledge_base_df["tail"] < n_items) & (knowledge_base_df["head"] >= n_items)]
    knowledge_base_inverse.columns = ["tail", "relation", "head"]

    ICM_df = pd.concat([knowledge_base_direct, knowledge_base_inverse], ignore_index=True)

    n_entities = max(ICM_df["head"].max(), ICM_df["tail"].max()) + 1
    ICM_sparse = sps.csr_matrix((np.ones(len(ICM_df), dtype=np.int),(ICM_df["head"].values, ICM_df["tail"].values)),
                          shape=(n_items, n_entities))

    ICM_sparse.data = np.ones_like(ICM_sparse.data)
    return ICM_sparse


class LastFM_KGAT_DataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, freeze_split = True):
        super(LastFM_KGAT_DataReader, self).__init__()

        pre_splitted_path = "result_experiments/{}/{}/".format("KGAT_data", "last-fm")
        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"
        
        dataset_dir = "SIGIR2022/KGAT_data/Data/last-fm/"
        
        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print("{}: Attempting to load saved data from {}".format(os.path.dirname(__file__), pre_splitted_path + pre_splitted_filename))
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)
                
        except FileNotFoundError:
            
            if freeze_split:
                raise Exception("Splitted data not found!")
            
            print("{}: Pre-splitted data not found, building new one".format(os.path.dirname(__file__)))
            print("{}: loading data".format(os.path.dirname(__file__)))

            URM_train_validation = self._load_data_file(os.path.join(dataset_dir, 'train.txt'))
            URM_test = self._load_data_file(os.path.join(dataset_dir, 'test.txt'))

            URM_train_validation.data = np.ones_like(URM_train_validation.data)
            URM_test.data = np.ones_like(URM_test.data)

            # Ensure the training data does not contain any test data
            URM_train_validation = URM_train_validation - URM_train_validation.multiply(URM_test)

            # Split user-wise
            URM_train, URM_validation = split_train_in_two_percentage_user_wise(URM_train_validation, train_percentage=0.90)


            self.knowledge_base_df = pd.read_csv(os.path.join(dataset_dir, 'kg_final.txt'), header=None, sep=" ")
            self.knowledge_base_df.columns = ["head", "relation", "tail"]
            self.knowledge_base_df.drop_duplicates()

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

            print("{}: loading complete".format(os.path.dirname(__file__)))


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