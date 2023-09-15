#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/06/2023

@author: Anonymized for blind review
"""

# Classe che prende i dati e li trasforma in matrici sparse

import json
import pickle
import re
import nltk
from collections import Counter
import pandas as pd
import random
import heapq
import csv
from tqdm import tqdm
import os
import numpy as np
import time
import math
# import lmdb
import gensim
import heapq

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
import torchfile
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
    GRAPH_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, freeze_split=False):
        super(RecipeRecDataReader, self).__init__()

        pre_splitted_path += "data_split/"
        print(pre_splitted_path)
        pre_splitted_filename = "splitted_data"

        base_ds_path = os.path.join(os.path.dirname(
            __file__), "../RecipeRec_github")
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

            graph_list, _ = load_graphs(pre_splitted_path +
                                        "graph_saved", idx_list=None)

            graph = graph_list[0].clone()
            print('Graph loaded from file graph: \n', graph)

            # Split train val test
            all_src_dst_weight, train_src_dst_weight, val_src_dst_weight, test_src_dst_weight = torch.load(
                dataset_dir+'/all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt')
            all_src, all_dst, all_weight = all_src_dst_weight
            train_src, train_dst, train_weight = train_src_dst_weight
            val_src, val_dst, val_weight = val_src_dst_weight
            test_src, test_dst, test_weight = test_src_dst_weight

            train_eids = graph.edge_ids(train_src, train_dst, etype='u-r')
            val_eids = graph.edge_ids(val_src, val_dst, etype='u-r')
            test_eids = graph.edge_ids(test_src, test_dst, etype='u-r')
            val_eids_r2u = graph.edge_ids(val_dst, val_src, etype='r-u')
            test_eids_r2u = graph.edge_ids(test_dst, test_src, etype='r-u')
            print('length of all_src: ', len(all_src))
            print('length of train_eids: ', len(train_eids))
            print('length of val_eids: ', len(val_eids))
            print('length of test_eids: ', len(test_eids))

            # get train_graph and val_graph
            train_graph = graph.clone()
            train_graph.remove_edges(
                torch.cat([val_eids, test_eids]), etype='u-r')
            train_graph.remove_edges(
                torch.cat([val_eids_r2u, test_eids_r2u]), etype='r-u')
            print('training graph: ')
            print(train_graph)
            print()

            val_graph = graph.clone()
            val_graph.remove_edges(test_eids, etype='u-r')
            val_graph.remove_edges(test_eids, etype='r-u')
            print('val graph: ')
            print(val_graph)

            print('generating edge dataloaders ...')
            # edge dataloaders
            sampler = dgl.dataloading.MultiLayerNeighborSampler([20, 20])
            neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)

            n_test_negs = 100  # number of negative recipes for each test user

            class test_NegativeSampler(object):

                def __init__(self, g, k):
                    # get the negatives
                    self.user2negs_100_dict = {}
                    filename = dataset_dir+'/test_negatives_100.txt'
                    with open(filename, "r") as f:
                        lines = f.readlines()
                        for line in tqdm(lines):
                            if line == None or line == "":
                                continue
                            line = line[:-1]  # remove \n
                            user = int(line.split('\t')[0].split(',')[0][1:])
                            negs = [int(neg) for neg in line.split('\t')[1:]]
                            self.user2negs_100_dict[user] = negs

                    self.k = k

                def __call__(self, g, eids_dict):
                    result_dict = {}
                    for etype, eids in eids_dict.items():
                        src, _ = g.find_edges(eids, etype=etype)
                        dst = []
                        for each_src in src:
                            dst.extend(
                                self.user2negs_100_dict[int(each_src)][:self.k])
                        dst = torch.tensor(dst)
                        src = src.repeat_interleave(self.k)
                        result_dict[etype] = (src, dst)
                    return result_dict

            test_neg_sampler = test_NegativeSampler(graph, n_test_negs)
            test_train_neg_sampler = test_NegativeSampler(graph, n_test_negs)

            train_collator = dgl.dataloading.EdgeCollator(
                train_graph, {
                    'u-r': train_graph.edge_ids(train_src, train_dst, etype='u-r')}, sampler,
                exclude='reverse_types',
                reverse_etypes={'u-r': 'r-u', 'r-u': 'u-r'},
                negative_sampler=neg_sampler)
            val_collator = dgl.dataloading.EdgeCollator(
                val_graph, {
                    'u-r': val_graph.edge_ids(val_src, val_dst, etype='u-r')}, sampler,
                exclude='reverse_types',
                reverse_etypes={'u-r': 'r-u', 'r-u': 'u-r'},
                negative_sampler=neg_sampler)
            test_collator = dgl.dataloading.EdgeCollator(
                graph, {('user', 'u-r', 'recipe'): test_eids}, sampler,
                exclude='reverse_types',
                reverse_etypes={'u-r': 'r-u', 'r-u': 'u-r'},
                negative_sampler=test_neg_sampler)

            train_edgeloader = torch.utils.data.DataLoader(
                train_collator.dataset, collate_fn=train_collator.collate,
                batch_size=1024, shuffle=True, drop_last=False, num_workers=0)
            val_edgeloader = torch.utils.data.DataLoader(
                val_collator.dataset, collate_fn=val_collator.collate,
                batch_size=128, shuffle=False, drop_last=False, num_workers=0)
            test_edgeloader = torch.utils.data.DataLoader(
                test_collator.dataset, collate_fn=test_collator.collate,
                batch_size=128, shuffle=False, drop_last=False, num_workers=0)

            print('# of batches in train_edgeloader: ', len(train_edgeloader))
            print('# of batches in val_edgeloader: ', len(val_edgeloader))
            print('# of batches in test_edgeloader: ', len(test_edgeloader))
            print()

            for input_nodes, pos_pair_graph, neg_pair_graph, blocks in train_edgeloader:
                print('blocks: ', blocks)
                break

            self.graph = graph
            self.train_graph = train_graph
            self.val_graph = val_graph
            self.train_edgeloader = train_edgeloader
            self.val_edgeloader = val_edgeloader
            self.test_edgeloader = test_edgeloader

        except FileNotFoundError:

            if freeze_split:
                raise Exception("Splitted data not found!")

            print(self.__class__.__name__ +
                  ": Pre-splitted data not found, building new one")
            print(self.__class__.__name__ + ": loading data")

            print('generating URMs ...')
            all_u2r_src_dst_weight, train_u2r_src_dst_weight, val_u2r_src_dst_weight, test_u2r_src_dst_weight = torch.load(
                dataset_dir+'/all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt')
            all_u_rate_r_edge_src, all_u_rate_r_edge_dst, all_u_rate_r_edge_weight = all_u2r_src_dst_weight
            train_u_rate_r_edge_src, train_u_rate_r_edge_dst, train_u_rate_r_edge_weight = train_u2r_src_dst_weight
            val_u_rate_r_edge_src, val_u_rate_r_edge_dst, val_u_rate_r_edge_weight = val_u2r_src_dst_weight
            test_u_rate_r_edge_src, test_u_rate_r_edge_dst, test_u_rate_r_edge_weight = test_u2r_src_dst_weight

            # Number of users: 7959 (c'è più di una interazione per ciascun utente)
            # Number of recipes: 68794
            # Number of ingredients: 8847
            # Number of interaction all u-r: 135353
            # Number of interaction all r-i: 463485

            n_users = 7959
            n_recipes = 68794
            n_ingredients = 8847

            # Il test e il validation set contengono un'interazione per ciascun utente
            # Sostituire l'array di weight con np.ones(135353) per avere una URM implicita

            URM_all = coo_matrix((all_u_rate_r_edge_weight, (
                all_u_rate_r_edge_src, all_u_rate_r_edge_dst)), shape=(n_users, n_recipes)).tocsr()
            URM_train = coo_matrix((train_u_rate_r_edge_weight, (
                train_u_rate_r_edge_src, train_u_rate_r_edge_dst)), shape=(n_users, n_recipes)).tocsr()
            URM_validation = coo_matrix((val_u_rate_r_edge_weight, (
                val_u_rate_r_edge_src, val_u_rate_r_edge_dst)), shape=(n_users, n_recipes)).tocsr()
            URM_test = coo_matrix((test_u_rate_r_edge_weight, (
                test_u_rate_r_edge_src, test_u_rate_r_edge_dst)), shape=(n_users, n_recipes)).tocsr()

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            print('generating ICM ...')
            edge_src, edge_dst, r_i_edge_weight = torch.load(
                dataset_dir+'/edge_r2i_src_dst_weight.pt')

            ICM = coo_matrix((r_i_edge_weight, (edge_src, edge_dst)),
                             shape=(n_recipes, n_ingredients)).tocsr()

            self.ICM_DICT = {
                "ICM_entities": ICM,
            }

            self.UCM_DICT = {}

            def get_graph():

                print('generating graph ...')
                edge_src, edge_dst, r_i_edge_weight = torch.load(
                    dataset_dir+'/edge_r2i_src_dst_weight.pt')
                recipe_edge_src, recipe_edge_dst, recipe_edge_weight = torch.load(
                    dataset_dir+'/edge_r2r_src_and_dst_and_weight.pt')
                ingre_edge_src, ingre_edge_dst, ingre_edge_weight = torch.load(
                    dataset_dir+'/edge_i2i_src_and_dst_and_weight.pt')
                all_u2r_src_dst_weight, train_u2r_src_dst_weight, val_u2r_src_dst_weight, test_u2r_src_dst_weight = torch.load(
                    dataset_dir+'/all_train_val_test_edge_u_rate_r_src_and_dst_and_weight.pt')
                u_rate_r_edge_src, u_rate_r_edge_dst, u_rate_r_edge_weight = all_u2r_src_dst_weight

                # nodes and edges
                graph = dgl.heterograph({
                    ('recipe', 'r-i', 'ingredient'): (edge_src, edge_dst),
                    ('ingredient', 'i-r', 'recipe'): (edge_dst, edge_src),
                    ('recipe', 'r-r', 'recipe'): (recipe_edge_src, recipe_edge_dst),
                    ('ingredient', 'i-i', 'ingredient'): (ingre_edge_src, ingre_edge_dst),
                    ('user', 'u-r', 'recipe'): (u_rate_r_edge_src, u_rate_r_edge_dst),
                    ('recipe', 'r-u', 'user'): (u_rate_r_edge_dst, u_rate_r_edge_src)
                })

                # edge weight
                graph.edges['r-i'].data['weight'] = torch.FloatTensor(
                    r_i_edge_weight)
                graph.edges['i-r'].data['weight'] = torch.FloatTensor(
                    r_i_edge_weight)
                graph.edges['r-r'].data['weight'] = torch.FloatTensor(
                    recipe_edge_weight)
                graph.edges['i-i'].data['weight'] = torch.FloatTensor(
                    ingre_edge_weight)
                graph.edges['u-r'].data['weight'] = torch.FloatTensor(
                    u_rate_r_edge_weight)
                graph.edges['r-u'].data['weight'] = torch.FloatTensor(
                    u_rate_r_edge_weight)

                # num of users: 7959
                # num of recipes: 68794
                # node features
                recipe_nodes_avg_instruction_features = torch.load(
                    dataset_dir+'/recipe_nodes_avg_instruction_features.pt')
                ingredient_nodes_nutrient_features_minus1 = torch.load(
                    dataset_dir+'/ingredient_nodes_nutrient_features.pt')
                graph.nodes['recipe'].data['avg_instr_feature'] = recipe_nodes_avg_instruction_features
                graph.nodes['ingredient'].data['nutrient_feature'] = ingredient_nodes_nutrient_features_minus1
                graph.nodes['user'].data['random_feature'] = torch.nn.init.xavier_normal_(
                    torch.ones(7959, 300))
                graph.nodes['recipe'].data['random_feature'] = torch.nn.init.xavier_normal_(
                    torch.ones(68794, 1024))

                return graph

            graph = get_graph()
            print('graph: ', graph)

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
                # "GRAPH_DICT": self.GRAPH_DICT,
            }

            save_graphs(pre_splitted_path + "graph_saved",
                        [graph])

            print('saving data in splitted_data.zip')
            dataIO.save_data(pre_splitted_filename,
                             data_dict_to_save=data_dict_to_save)

            print(self.__class__.__name__ + ": loading complete")
