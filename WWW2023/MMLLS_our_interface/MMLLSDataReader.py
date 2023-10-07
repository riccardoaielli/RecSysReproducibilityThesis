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
from dgl.nn.functional import edge_softmax
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.data.graph_serialize import *

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


class RecipeRecDataReader(object):

    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split=False):
        super(RecipeRecDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        print(pre_splitted_path)
        pre_splitted_filename = "splitted_data"

        base_ds_path = os.path.join(os.path.dirname(
            __file__), "../RecipeRec_github")  # TODO
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

            # TODO Fai la load da file
            # TODO setta tutti i valori nll'oggetto
            self.graph = graph
            self.train_graph = train_graph
            self.val_graph = val_graph
            self.train_edgeloader = train_edgeloader
            self.val_edgeloader = val_edgeloader
            self.test_edgeloader = test_edgeloader
            self.n_test_negs = n_test_negs

        except FileNotFoundError:

            if freeze_split:
                raise Exception("Splitted data not found!")

            print(self.__class__.__name__ +
                  ": Pre-splitted data not found, building new one")
            print(self.__class__.__name__ + ": loading data")

            # TODO FAi la store/load da file

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
