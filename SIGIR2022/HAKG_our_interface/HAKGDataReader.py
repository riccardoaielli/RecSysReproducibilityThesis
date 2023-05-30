#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from Recommenders.DataIO import DataIO
import scipy.sparse as sps
import os, shutil, zipfile
import pandas as pd
import numpy as np

from SIGIR2022.KGAT_data.LastFM_KGAT_DataReader import _get_ICM_from_df


class HAKGDataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split = True):
        super(HAKGDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

        original_data_path = "SIGIR2022/HAKG_github/data/"

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

            if dataset_name == "yelp2018":
                zipFile_path = "{}/{}.zip".format(original_data_path, dataset_name)

                dataFile = zipfile.ZipFile(zipFile_path)
                URM_train_path = dataFile.extract(dataset_name + '/train.txt', path=zipFile_path + "decompressed/")
                URM_test_path = dataFile.extract(dataset_name + '/test.txt', path=zipFile_path + "decompressed/")
                KG_path = dataFile.extract(dataset_name + '/kg_final.txt', path=zipFile_path + "decompressed/")

                URM_train_validation = self._load_data_file(URM_train_path)
                URM_test = self._load_data_file(URM_test_path)

                self.knowledge_base_df = pd.read_csv(KG_path, header=None, sep=" ")

                shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

            elif dataset_name == "alibaba-ifashion":
                original_data_path = os.path.join(original_data_path, "alibaba-ifashion")

                URM_train_validation = self._load_data_file(os.path.join(original_data_path, 'train.txt'))
                URM_test = self._load_data_file(os.path.join(original_data_path, 'test.txt'))
                self.knowledge_base_df = pd.read_csv(os.path.join(original_data_path, 'kg_final.txt'), header=None, sep=" ")

            else:
                raise ValueError()

            self.knowledge_base_df.columns = ["head", "relation", "tail"]
            self.knowledge_base_df.drop_duplicates()

            # Split user-wise
            URM_train, URM_validation = split_train_in_two_percentage_user_wise(URM_train_validation, train_percentage=0.89)

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



class HAKG_Add_KB_to_existing_split_DataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, collaborative_dataset, pre_splitted_path, freeze_split=True):
        super(HAKG_Add_KB_to_existing_split_DataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"

        original_data_path = "SIGIR2022/HAKG_github/data/"

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

            if dataset_name == "yelp2018":
                zipFile_path = "{}/{}.zip".format(original_data_path, dataset_name)

                dataFile = zipfile.ZipFile(zipFile_path)
                KG_path = dataFile.extract(dataset_name + '/kg_final.txt', path=zipFile_path + "decompressed/")

                self.knowledge_base_df = pd.read_csv(KG_path, header=None, sep=" ")

                shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

            else:
                raise ValueError()

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













