#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import  split_train_in_two_percentage_global_sample
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.Yelp.YelpReader import YelpReader, Yelp2022Reader
from Data_manager.AmazonReviewData.AmazonBooksReader import AmazonBooksReader
from Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores
from Data_manager.DataPostprocessing_Implicit_URM import DataPostprocessing_Implicit_URM

import os

from Recommenders.DataIO import DataIO


class RGCFDataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split = True):
        super(RGCFDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

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

            # All datasets are transformed to implicit setting interactions >= 4 to 1 and the others to 0
            # Ensure that each user and item of yelp and amazonbook has at least 15 interactions applying k-core postprocessing
            # k-core is iterative and stops only at convergence when the condition of >= 15 is true for all users and items
            if dataset_name == 'movielens1m':
                data_reader = Movielens1MReader()
                # The paper says to select only interactions >= 4 but the statistics reported in the paper correspond to >=3
                data_reader = DataPostprocessing_Implicit_URM(data_reader, greater_equal_than=3)

            elif dataset_name == 'yelp':
                data_reader = Yelp2022Reader()
                # The paper says to select only interactions >= 4 but the statistics reported in the paper are similar to what obtained
                # by retaining all interactions
                data_reader = DataPostprocessing_Implicit_URM(data_reader, greater_equal_than=0)
                data_reader = DataPostprocessing_K_Cores(data_reader, k_cores_value=15)

            elif dataset_name == 'amazon-book':
                data_reader = AmazonBooksReader()
                # The paper says to select only interactions >= 4 but the statistics reported in the paper correspond to >=3
                data_reader = DataPostprocessing_Implicit_URM(data_reader, greater_equal_than=3)
                data_reader = DataPostprocessing_K_Cores(data_reader, k_cores_value=15)

            else:
                raise ValueError("Dataset name not supported, current is {}".format(dataset_name))

            loaded_dataset = data_reader.load_data()

            URM_all = loaded_dataset.AVAILABLE_URM['URM_all']

            # Split is 80% training, 10% validation and 10% test
            URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.90)
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.89)

            self.ICM_DICT = loaded_dataset.AVAILABLE_ICM
            self.UCM_DICT = loaded_dataset.AVAILABLE_UCM

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
