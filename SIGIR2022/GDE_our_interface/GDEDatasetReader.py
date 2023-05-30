#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2022

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

import os
import numpy as np
import pandas as pd
import scipy.sparse as sps

from Recommenders.DataIO import DataIO


class GDEDatasetReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split = True):
        super(GDEDatasetReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

        base_ds_path = os.path.join(os.path.dirname(__file__), "../GDE_github/datasets")
        dataset_dir = os.path.join(base_ds_path, dataset_name)

        if not os.path.exists(dataset_dir):
            raise ValueError('Invalid dataset %s' % dataset_name)

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print(self.__class__.__name__ + ": Attempting to load saved data from " + pre_splitted_path + pre_splitted_filename)
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)

        except FileNotFoundError:
            
            if freeze_split:
                raise Exception("Splitted data not found!")
            
            print(self.__class__.__name__ + ": Pre-splitted data not found, building new one")
            print(self.__class__.__name__ + ": loading data")

            ds_train = pd.read_csv(os.path.join(dataset_dir, 'train_sparse.csv')).drop_duplicates()
            ds_test = pd.read_csv(os.path.join(dataset_dir, 'test_sparse.csv')).drop_duplicates()

            n_users = max(ds_train['user'].max(), ds_test['user'].max()) + 1
            n_items = max(ds_train['item'].max(), ds_test['item'].max()) + 1

            shape = (n_users, n_items)

            intersection_dataframe = pd.merge(ds_train, ds_test, on=['user', 'item'], how='inner')
            if not intersection_dataframe.empty:
                print("Dataset: {} contains {} interactions that appear both in the training and test data. Those interactions will be removed from the training data".format(dataset_name, len(intersection_dataframe)))

                # Find interactions present in both datasets and remove them from the training set
                ds_train = pd.merge(ds_train, ds_test, indicator=True, how='outer')\
                   .query('_merge=="left_only"')\
                   .drop('_merge', axis=1)

            assert pd.merge(ds_train, ds_test, on=['user', 'item'], how='inner').empty

            URM_train = self.build_urm(ds_train, shape)
            URM_test = self.build_urm(ds_test, shape)

            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train.copy(), train_percentage=0.95)

            self.ICM_DICT = {}
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
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print(self.__class__.__name__ + ": loading complete")

    def build_urm(self, ds, shape):

        return sps.csr_matrix((
            np.ones(ds['user'].size).astype(int),
            (ds['user'].values.astype(int),
             ds['item'].values.astype(int)),
        ), shape=shape)
