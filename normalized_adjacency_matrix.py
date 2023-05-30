#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2023

@author: Anonymized for blind review
"""

import torch
import numpy as np
import scipy.sparse as sps

def normalized_adjacency_matrix(URM, add_self_connection = False):

    n_users, n_items = URM.shape

    Zero_u_u_sps = sps.csr_matrix((n_users, n_users))
    Zero_i_i_sps = sps.csr_matrix((n_items, n_items))
    A = sps.bmat([[Zero_u_u_sps,     URM ],
                  [URM.T, Zero_i_i_sps   ]], format="csr")

    if add_self_connection:
        A = A + sps.eye(A.shape[0])

    D_inv = 1/(np.sqrt(np.array(A.sum(axis = 1)).squeeze()) + 1e-6)
    A_tilde = sps.diags(D_inv).dot(A).dot(sps.diags(D_inv)).astype(np.float32)

    return A_tilde


def from_sparse_to_tensor(A_tilde):
    A_tilde = sps.coo_matrix(A_tilde)
    A_tilde = torch.sparse_coo_tensor(np.vstack([A_tilde.row, A_tilde.col]), A_tilde.data, A_tilde.shape)
    A_tilde = A_tilde.coalesce()

    return A_tilde







