#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21/01/2023

@author: Anonymized for blind review
"""
import pandas as pd

from SIGIR2022.GDE_our_interface.GDEDatasetReader import GDEDatasetReader
from SIGIR2022.HCCF_our_interface.HCCFDataReader import HCCFDataReader
from SIGIR2022.INMO_our_interface.INMODataReader import INMODataReader
from SIGIR2022.RGCF_our_interface.RGCFDataReader import RGCFDataReader
from SIGIR2022.SimGCL_our_interface.SimGCLDataReader import SimGCLDataReader
from SIGIR2022.LightGCN_our_interface.LightGCNDataReader import LightGCNReader
from SIGIR2022.HAKG_our_interface.HAKGDataReader import HAKGDataReader
from SIGIR2022.KGCL_our_interface.KGCLDataReader import KGCLDataReader
from SIGIR2022.KGAT_data.LastFM_KGAT_DataReader import LastFM_KGAT_DataReader

def _is_equals(URM_train_1, URM_test_1, URM_train_2, URM_test_2):

    if not URM_train_1.shape == URM_test_2.shape:
        return False

    if not (URM_train_1 + URM_test_1).nnz == (URM_train_2 + URM_test_2).nnz:
        return False

    if not URM_train_1.nnz == URM_train_2.nnz or not URM_test_1.nnz == URM_test_2.nnz:
        return False

    return (URM_train_1 - URM_train_2).nnz == 0 and (URM_test_1 - URM_test_2).nnz == 0


def _is_equals_dataset(dataset_1, dataset_2):

    return _is_equals(
        dataset_1.URM_DICT["URM_train"] + dataset_1.URM_DICT["URM_validation"],
        dataset_1.URM_DICT["URM_test"],
        dataset_2.URM_DICT["URM_train"] + dataset_2.URM_DICT["URM_validation"],
        dataset_2.URM_DICT["URM_test"],
    )

import itertools

result_folder_path = "result_experiments"# + "/{}/{}_{}/".format(ALGORITHM_NAME, dataset_name, split_type)

dataset_dict_all = {
    "movielens100k": {
            "GDE": GDEDatasetReader("ml_100k", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "GDE", "movielens100k"))
        },

    "movielens1m": {
            "GDE": GDEDatasetReader("ml_1m", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "GDE", "movielens1m")),
            "RGCF": RGCFDataReader("movielens1m", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "RGCF", "movielens1m")),
        },

    "movielens10m": {
            "HCCF": HCCFDataReader("ml10m", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "HCCF", "movielens10m"))
        },

    "citeulike": {
            "GDE": GDEDatasetReader("citeulike", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "GDE", "citeulike"))
        },

    "pinterest": {
            "GDE": GDEDatasetReader("pinterest", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "GDE", "pinterest"))
        },

    "gowalla": {
            "GDE": GDEDatasetReader("gowalla", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "GDE", "gowalla")),
            "INMO": INMODataReader("Gowalla", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "INMO", "gowalla"), fold=0),
            "LightGCN": LightGCNReader("gowalla", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "LightGCN", "gowalla", "original"), split_type="original"),
            "GTN": LightGCNReader("gowalla", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "LightGCN", "gowalla", "original"), split_type="original"),
        },

    "amazon-book": {
            "HCCF": HCCFDataReader("amazon", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "HCCF", "amazon-book")),
            "INMO": INMODataReader("Amazon", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "INMO", "amazon-book"), fold=0),
            "RGCF": RGCFDataReader("amazon-book", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "RGCF", "amazon-book")),
            "SimGCL": SimGCLDataReader("{}/{}/{}_{}/data/".format(result_folder_path, "SimGCL", "amazon-book", "original"), "amazon-book", "original"),
            "LightGCN": LightGCNReader("amazon-book", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "LightGCN", "amazon-book", "original"), split_type="original"),
            "GTN": LightGCNReader("amazon-book", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "LightGCN", "amazon-book", "original"), split_type="original"),
            "KGCL": KGCLDataReader("amazon-book", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "KGCL", "amazon-book", "original"), split_type="original"),
        },

    "yelp2018": {
            "HCCF": HCCFDataReader("yelp", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "HCCF", "yelp2018")),
            "INMO": INMODataReader("Yelp", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "INMO", "yelp2018"), fold=0),
            "RGCF": RGCFDataReader("yelp", pre_splitted_path = "{}/{}/{}/data/".format(result_folder_path, "RGCF", "yelp")),
            "SimGCL": SimGCLDataReader("{}/{}/{}_{}/data/".format(result_folder_path, "SimGCL", "yelp2018", "original"), "yelp2018", "original"),
            "LightGCN": LightGCNReader("yelp2018", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "LightGCN", "yelp2018", "original"), split_type="original"),
            "GTN": LightGCNReader("yelp2018", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "LightGCN", "yelp2018", "original"), split_type="original"),
            "HAKG": HAKGDataReader("yelp2018", "{}/{}/{}_{}/data/".format(result_folder_path, "HAKG", "yelp2018", "original")),
            "KGCL": KGCLDataReader("yelp2018", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "KGCL", "yelp2018", "original"), split_type="original"),
        },

    "doubanbook": {
            "SimGCL": SimGCLDataReader("{}/{}/{}_{}/data/".format(result_folder_path, "SimGCL", "doubanbook", "original"), "doubanbook", "original")
        },

    "MIND": {
            "KGCL": KGCLDataReader("MIND", pre_splitted_path = "{}/{}/{}_{}/data/".format(result_folder_path, "KGCL", "MIND", "original"), split_type="original"),
    },

    "last-fm": {
            "GTN": LastFM_KGAT_DataReader(),
            "HAKG": LastFM_KGAT_DataReader(),
    },

    "alibaba-ifashion": {
            "HAKG": HAKGDataReader("alibaba-ifashion", "{}/{}/{}_{}/data/".format(result_folder_path, "HAKG", "alibaba-ifashion", "original"))
    },
}


dataset_list = dataset_dict_all.keys()

print("\n\n\n\n\n\n\n")

print_only_for = "KGCL"

for dataset_name in dataset_list:

    this_dataset_dict = dataset_dict_all[dataset_name]

    if len(this_dataset_dict)>1:
        label_list = list(this_dataset_dict.keys())
        label_list.sort()

        comparison_df = pd.DataFrame(index=label_list, columns=label_list)

        for label1, label2 in itertools.product(this_dataset_dict.keys(), this_dataset_dict.keys()):
            if label2<=label1:
                continue

            # if not(label1 == print_only_for or label2 == print_only_for):
            #     continue

            if _is_equals_dataset(this_dataset_dict[label1], this_dataset_dict[label2]):
                print("On Dataset {}, {} and {} are equal.".format(dataset_name, label1, label2))
                comparison_df.loc[label1, label2] = True
                # comparison_df.loc[label2, label1] = None
            else:
                comparison_df.loc[label1, label2] = False
                # comparison_df.loc[label2, label1] = None

        comparison_df.to_csv("result_experiments/comparison_dataset_{}.csv".format(dataset_name))
        comparison_df.to_latex("result_experiments/comparison_dataset_{}.tex".format(dataset_name), na_rep="-")




