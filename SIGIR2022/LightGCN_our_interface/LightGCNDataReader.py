#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import os
from Recommenders.DataIO import DataIO


class LightGCNReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, split_type, freeze_split = True):
        super(LightGCNReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"
        
        dataset_dir = os.path.join("SIGIR2022/LightGCN_github/data/", dataset_name)
        
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

            URM_train_validation = self._load_data_file(os.path.join(dataset_dir, 'train.txt'))
            URM_test = self._load_data_file(os.path.join(dataset_dir, 'test.txt'))

            # Split is 72% training, 8% validation and 20% test
            if split_type == "ours":
                URM_all = URM_train_validation + URM_test
                URM_train_validation, URM_test = split_train_in_two_percentage_user_wise(URM_all, train_percentage=0.80)

            # Split user-wise
            URM_train, URM_validation = split_train_in_two_percentage_user_wise(URM_train_validation, train_percentage=0.90)

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