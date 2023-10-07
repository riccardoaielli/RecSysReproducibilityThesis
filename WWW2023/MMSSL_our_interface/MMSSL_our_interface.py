from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import dgl
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd
import copy
from torch.utils.tensorboard import SummaryWriter


def weights_init(self, m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def csr_norm(self, csr_mat, mean_flag=False):
    rowsum = np.array(csr_mat.sum(1))
    rowsum = np.power(rowsum+1e-8, -0.5).flatten()
    rowsum[np.isinf(rowsum)] = 0.
    rowsum_diag = sp.diags(rowsum)

    colsum = np.array(csr_mat.sum(0))
    colsum = np.power(colsum+1e-8, -0.5).flatten()
    colsum[np.isinf(colsum)] = 0.
    colsum_diag = sp.diags(colsum)

    if mean_flag == False:
        return rowsum_diag*csr_mat*colsum_diag
    else:
        return rowsum_diag*csr_mat


def matrix_to_tensor(self, cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  #
    indices = torch.from_numpy(
        np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
    values = torch.from_numpy(cur_matrix.data)  #
    shape = torch.Size(cur_matrix.shape)

    #
    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(config['device'])


def bpr_loss(self, users, pos_items, neg_items):
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

    regularizer = 1./2*(users**2).sum() + 1./2 * \
        (pos_items**2).sum() + 1./2*(neg_items**2).sum()
    regularizer = regularizer / self.batch_size

    maxi = F.logsigmoid(pos_scores - neg_scores)
    mf_loss = -torch.mean(maxi)

    emb_loss = self.decay * regularizer
    reg_loss = 0.0
    return mf_loss, emb_loss, reg_loss


def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def u_sim_calculation(self, users, user_final, item_final):
    topk_u = user_final[users]
    u_ui = torch.tensor(self.ui_graph_raw[users].todense()).to(
        config['device'])

    num_batches = (self.n_items - 1) // args.batch_size + 1
    indices = torch.arange(0, self.n_items).to(config['device'])
    u_sim_list = []

    for i_b in range(num_batches):
        index = indices[i_b * args.batch_size:(i_b + 1) * args.batch_size]
        sim = torch.mm(topk_u, item_final[index].T)
        sim_gt = torch.multiply(sim, (1-u_ui[:, index]))
        u_sim_list.append(sim_gt)

    u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
    return u_sim
