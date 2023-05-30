#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/2022

@author: Anonymized for blind review
"""

import pandas as pd



def save_data_statistics(URM_all, dataset_name, output_file_path):

    n_users, n_items = URM_all.shape

    statistics_df = pd.DataFrame(index=[dataset_name])
    statistics_df.loc[dataset_name, "Interactions"] = URM_all.nnz
    statistics_df.loc[dataset_name, "Items"] = n_items
    statistics_df.loc[dataset_name, "Users"] = n_users
    statistics_df.loc[dataset_name, "Sparsity"] = "{:.3E}".format(1-URM_all.nnz/(n_items*n_users))

    statistics_df.to_latex(output_file_path, float_format="{:0.0f}".format)




