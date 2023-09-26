#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2022

@author: Anonymized for blind review
"""

# Classe che prende i dati e li trasforma in matrici sparse

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

import os
import numpy as np
import pickle
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import scipy.sparse as sp
from WWW2023.AutoCF_our_interface.AutoCF_our_interface import log
import torch as t
from Recommenders.DataIO import DataIO


class AutoCFDataReader(object):
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def __init__(self, dataset_name, freeze_split=True):
        super(AutoCFDataReader, self).__init__()

        base_ds_path = os.path.join(os.path.dirname(
            __file__), "../AutoCF_github/Datasets")
        dataset_dir = os.path.join(base_ds_path, dataset_name)

        if not os.path.exists(dataset_dir):
            raise ValueError('Invalid dataset %s' % dataset_name)

        self.trnfile = os.path.join(dataset_dir, 'trnMat.pkl')
        self.tstfile = os.path.join(dataset_dir, 'tstMat.pkl')
        self.valfile = os.path.join(dataset_dir, 'valMat.pkl')

        print(self.__class__.__name__ + ": Attempting to load data from " +
              dataset_dir)

        # Mettere tutto quello che serve a caricare i dati e salva tutto in self.

        self.trnMat = self.loadOneFile(self.trnfile)
        self.tstMat = self.loadOneFile(self.tstfile)
        self.valMat = self.loadOneFile(self.valfile)

        URM_train = self.trnMat.tocsr()
        URM_test = self.tstMat.tocsr()
        URM_validation = self.valMat.tocsr()

        """ # Riduco la dimensione a 100 utenti per fare testing
        URM_train = URM_train[0:100, :]
        URM_test = URM_test[0:100, :]
        URM_validation = URM_validation[0:100, :]

        self.trnMat = URM_train.tocoo()
        self.tstMat = URM_test.tocoo()
        self.valMat = URM_validation.tocoo()
        print(URM_train.todense())
        ########################################### """

        self.ICM_DICT = {}
        self.UCM_DICT = {}

        self.URM_DICT = {
            "URM_train": URM_train,
            "URM_test": URM_test,
            "URM_validation": URM_validation,
        }

        print(self.__class__.__name__ + ": loading complete")
