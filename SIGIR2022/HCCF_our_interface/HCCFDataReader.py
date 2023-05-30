#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/12/2022

@author: Anonymized for blind review
"""

import os, pickle
import scipy.sparse as sps
import numpy as np
from Recommenders.DataIO import DataIO


class HCCFDataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split = True):
        super(HCCFDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"
        
        dataset_dir = os.path.join("SIGIR2022/HCCF_github/Data/", dataset_name)
        
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
            
            print("{}: Pre-splitted data not found, building new one. Please ensure to decompress the original RAR files.".format(dataset_name))
            print("{}: loading data".format(dataset_name))
            #
            #
            # for split in ['trn', 'val', 'tst']:
            #     with open(dataset_dir + f"{split}Mat.pkl", 'rb') as fs:
            #         (pickle.load(fs) != 0).astype(np.float32)

            def _load_file(filename):
                coo_matrix = pickle.load(open(filename, 'rb'))
                coo_matrix.eliminate_zeros()
                coo_matrix.data = np.ones_like(coo_matrix.data)
                return sps.csr_matrix(coo_matrix, dtype=np.float32)

            URM_train = _load_file(dataset_dir + "/trnMat.pkl")
            URM_validation = _load_file(dataset_dir + "/valMat.pkl")
            URM_test = _load_file(dataset_dir + "/tstMat.pkl")

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
