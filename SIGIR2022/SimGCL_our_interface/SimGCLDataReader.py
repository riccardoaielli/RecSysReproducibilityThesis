#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import  split_train_in_two_percentage_global_sample

import os
import numpy as np
import scipy.sparse as sps

from Recommenders.DataIO import DataIO


class SimGCLDataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, pre_splitted_path, dataset_name, split_type, freeze_split = True):
        super(SimGCLDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

        if dataset_name == 'doubanbook':
            dataset_dir = os.path.join("SIGIR2022/SimGCL_torch_github/dataset/douban-book")

        elif dataset_name == 'yelp2018':
            dataset_dir = os.path.join("SIGIR2022/SimGCL_torch_github/dataset/yelp2018")

        elif dataset_name == 'amazon-book':
            dataset_dir = os.path.join("SIGIR2022/SimGCL_our_interface/amazonbook")

        else:
            raise ValueError("Dataset name not supported, current is {}".format(dataset_name))

        
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
            
            data_array = np.loadtxt(os.path.join(dataset_dir, 'train.txt')).astype(int)
            URM_train = sps.csr_matrix((np.ones_like(data_array[:, 2]), (data_array[:, 0], data_array[:, 1])))
            
            data_array = np.loadtxt(os.path.join(dataset_dir, 'test.txt')).astype(int)
            URM_test = sps.csr_matrix((np.ones_like(data_array[:, 2]), (data_array[:, 0], data_array[:, 1])))

            # Split is 70% training, 10% validation and 20% test
            if split_type == "ours":
                URM_all = URM_train + URM_test
                URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)


            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.875)


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
