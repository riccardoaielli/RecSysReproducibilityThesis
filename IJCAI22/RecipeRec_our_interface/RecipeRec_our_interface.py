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


from functools import partial
from dgl._ffi.base import DGLError

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
# from dgl.nn import EdgeWeightNorm

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

from IJCAI22.RecipeRec_our_interface.RecipeRecDataReader import RecipeRecDataReader


####### SET TRANSFORMER #######


def find(tensor, values):
    return torch.nonzero(tensor[..., None] == values)

# example of find()
# a = torch.tensor([0, 10, 20, 30])
# b = torch.tensor([[ 0, 30, 20,  10, 77733],[ 0, 30, 20,  10, 77733]])
# find(b, a)[:, 2]


def get_ingredient_neighbors_all_embeddings(blocks, output_nodes, secondToLast_ingre, device, ingre_neighbor_tensor, ingre_length_tensor):
    ingreNodeIDs = blocks[1].srcdata['_ID']['ingredient']
    recipeNodeIDs = output_nodes
    batch_ingre_neighbors = ingre_neighbor_tensor[recipeNodeIDs].to(device)
    batch_ingre_length = ingre_length_tensor[recipeNodeIDs]
    valid_batch_ingre_neighbors = find(
        batch_ingre_neighbors, ingreNodeIDs)[:, 2]

    # based on valid_batch_ingre_neighbors each row index
    _, valid_batch_ingre_length = torch.unique(
        find(batch_ingre_neighbors, ingreNodeIDs)[:, 0], return_counts=True)
    batch_sum_ingre_length = np.cumsum(valid_batch_ingre_length.cpu())

    total_ingre_emb = None
    for i in range(len(recipeNodeIDs)):
        if i == 0:
            recipeNode_ingres = valid_batch_ingre_neighbors[0:batch_sum_ingre_length[i]]
            a = secondToLast_ingre[recipeNode_ingres]
        else:
            recipeNode_ingres = valid_batch_ingre_neighbors[batch_sum_ingre_length[i-1]                                                            :batch_sum_ingre_length[i]]
            a = secondToLast_ingre[recipeNode_ingres]

        # all ingre instead of average
        a_rows = a.shape[0]
        a_columns = a.shape[1]
        max_rows = 5
        if a_rows < max_rows:
            a = torch.cat(
                [a, torch.zeros(max_rows-a_rows, a_columns).to(device)])
        else:
            a = a[:max_rows, :]

        if total_ingre_emb == None:
            total_ingre_emb = a.unsqueeze(0)
        else:
            total_ingre_emb = torch.cat(
                [total_ingre_emb, a.unsqueeze(0)], dim=0)
            if torch.isnan(total_ingre_emb).any():
                print('Error!')

    return total_ingre_emb


class Attention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature):
        super().__init__()

        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys, values):
        """
        It is equivariant to permutations
        of the batch dimension (`b`).

        It is equivariant to permutations of the
        second dimension of the queries (`n`).

        It is invariant to permutations of the
        second dimension of keys and values (`m`).

        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d'].
        Returns:
            a float tensor with shape [b, n, d'].
        """

        attention = torch.bmm(queries, keys.transpose(1, 2))
        attention = self.softmax(attention / self.temperature)
        # it has shape [b, n, m]

        return torch.bmm(attention, values)


class MultiheadAttention(nn.Module):

    def __init__(self, d, h):
        """
        Arguments:
            d: an integer, dimension of queries and values.
                It is assumed that input and
                output dimensions are the same.
            h: an integer, number of heads.
        """
        super().__init__()

        assert d % h == 0
        self.h = h

        # everything is projected to this dimension
        p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        self.attention = Attention(temperature=p**0.5)

    def forward(self, queries, keys, values):
        """
        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]

        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)

        queries = queries.permute(
            2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        # shape [h * b, n, p]
        output = self.attention(queries, keys, values)
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        return output


class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """

    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)


class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()

        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).

        It is invariant to permutations of the
        second dimension of tensor y (`m`).

        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))


class SetAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)


class InducedSetAttentionBlock(nn.Module):

    def __init__(self, d, m, h, rff1, rff2):
        """
        Arguments:
            d: an integer, input dimension.
            m: an integer, number of inducing points.
            h: an integer, number of heads.
            rff1, rff2: modules, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, rff1)
        self.mab2 = MultiheadAttentionBlock(d, h, rff2)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = x.size(0)
        p = self.inducing_points
        p = p.repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(p, x)  # shape [b, m, d]
        return self.mab2(x, h)


class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d].
        """
        b = z.size(0)
        s = self.seed_vectors
        s = s.repeat([b, 1, 1])  # random seed vector: shape [b, k, d]

        output = self.mab(s, z)
        # print('PoolingMultiheadAttention', output.shape)

        return output

# Set transformer for ingredient representation


class SetTransformer(nn.Module):
    def __init__(self, drop_out):
        """
        Arguments:
            in_dimension: an integer.  # 2
            out_dimension: an integer. # 5 * K
        """
        super(SetTransformer, self).__init__()
        in_dimension = 46  # 300
        out_dimension = 128  # 600 TODO è la embedding size???

        d = in_dimension
        m = 46  # number of inducing points
        h = 2  # 4 # number of heads
        k = 4  # number of seed vectors

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )

        self.decoder_2 = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d))
        )
        self.decoder_3 = nn.Sequential(
            SetAttentionBlock(d, h, RFF(d))
        )

        self.predictor = nn.Sequential(
            nn.Linear(k * d, out_dimension),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        x = self.encoder(
            x)  # x = self.encoder(cut_x) # shape [batch, batch_max_len, d]
        x = self.dropout(x)
        x = self.decoder(x)  # shape [batch, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)

        y = self.predictor(x)
        return y

###### GNN ######


class custom_HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(custom_HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(
                v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(
                    k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype], mod_kwargs),
                    *mod_args.get(etype, ())
                )
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype], mod_kwargs),
                    *mod_args.get(etype, ())
                )
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]


def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]


def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)


def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)


def _stack_agg_func(inputs, dsttype):
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)


def _agg_func(inputs, dsttype, fn):
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)


def get_aggregate_fn(agg):
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'), etype='u-r')
            return edge_subgraph.edata['score'][('user', 'u-r', 'recipe')].squeeze()


class RelationAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(RelationAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out = (beta * z).sum(1)                        # (N, D * K)
        return out


def node_drop(feats, drop_rate, training):
    n = feats.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats / (1. - drop_rate)
    else:
        feats = feats
    return feats


class custom_GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.1,
                 attn_drop=0.,
                 negative_slope=0.2,
                 edge_drop=0.1,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(custom_GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
            self.fc_src2 = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst2 = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc2 = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_l2 = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r2 = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

        self.edge_drop = edge_drop

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_l2, gain=gain)
        nn.init.xavier_normal_(self.attn_r2, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                do_edge_drop = feat[2]
                # print('do_edge_drop: ', do_edge_drop)
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                h_src2 = h_src.clone()
                h_dst2 = h_dst.clone()
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(
                        h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(
                        h_dst).view(-1, self._num_heads, self._out_feats)
                    feat_src2 = self.fc2(
                        h_src2).view(-1, self._num_heads, self._out_feats)
                    feat_dst2 = self.fc2(
                        h_dst2).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(
                        h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(
                        h_dst).view(-1, self._num_heads, self._out_feats)
                    feat_src2 = self.fc_src2(
                        h_src2).view(-1, self._num_heads, self._out_feats)
                    feat_dst2 = self.fc_dst2(
                        h_dst2).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                h_src2 = h_dst2 = h_src.clone()  # self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                feat_src2 = feat_dst2 = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    feat_dst2 = feat_src2[:graph.number_of_dst_nodes()]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph.srcdata.update(
                {'ft': feat_src, 'el': el, 'feat_src2': feat_src2})
            graph.dstdata.update({'er': er, 'feat_dst2': feat_dst2})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))

            # compute softmax, edge dropout
            if self.training and do_edge_drop and self.edge_drop > 0:
                perm = torch.randperm(
                    graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(
                    edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'initial_ft'))
            graph.update_all(fn.u_mul_v('feat_src2', 'feat_dst2', 'm2'),
                             fn.sum('m2', 'add_ft'))
            rst = graph.dstdata['initial_ft'] + graph.dstdata['add_ft']

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + \
                    self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class GNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, attentions_heads):
        super().__init__()

        self.num_heads = attentions_heads  # 8
        self.hid_feats = int(hid_feats/self.num_heads)
        self.out_feats = int(out_feats/self.num_heads)
        self.relation_attention = RelationAttention(hid_feats)

        self.gatconv1 = custom_HeteroGraphConv({  # dglnn.HeteroGraphConv
            'i-r': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'r-i': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'r-r': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'i-i': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'u-r': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'r-u': custom_GATConv(in_feats, self.hid_feats, num_heads=self.num_heads),
        }, aggregate='stack')

        self.gatconv2 = custom_HeteroGraphConv({  # dglnn.HeteroGraphConv
            'i-r': custom_GATConv(self.hid_feats*self.num_heads, self.out_feats, num_heads=self.num_heads),
            'r-i': custom_GATConv(self.hid_feats*self.num_heads, self.out_feats, num_heads=self.num_heads),
            'r-r': custom_GATConv(self.hid_feats*self.num_heads, self.out_feats, num_heads=self.num_heads),
            'i-i': custom_GATConv(self.hid_feats*self.num_heads, self.out_feats, num_heads=self.num_heads),
            'u-r': custom_GATConv(self.hid_feats*self.num_heads, self.out_feats, num_heads=self.num_heads),
            'r-u': custom_GATConv(self.hid_feats*self.num_heads, self.out_feats, num_heads=self.num_heads),
        }, aggregate='stack')

        self.dropout = nn.Dropout(0.1)

    def forward(self, blocks, inputs, do_edge_drop):
        edge_weight_0 = blocks[0].edata['weight']
        edge_weight_1 = blocks[1].edata['weight']

        num_users = blocks[-1].dstdata[dgl.NID]['user'].shape[0]
        num_recipes = blocks[-1].dstdata[dgl.NID]['recipe'].shape[0]

        h = self.gatconv1(blocks[0], inputs, edge_weight_0, do_edge_drop)
        h = {k: F.relu(v).flatten(2) for k, v in h.items()}
        h = {k: self.relation_attention(v) for k, v in h.items()}

        first_layer_output = {}
        first_layer_output['user'] = h['user'][:num_users]
        first_layer_output['recipe'] = h['recipe'][:num_recipes]

        h = {key: self.dropout(value) for key, value in h.items()}
        h = self.gatconv2(blocks[-1], h, edge_weight_1, do_edge_drop)
        last_ingre_and_instr = h['recipe'].flatten(2)
        h = {k: self.relation_attention(v.flatten(2))
             for k, v in h.items()}

        return h

#         # combine several layer embs as the final emb
#         combined_output = {}
#         combined_output['user'] = torch.cat([h['user'], first_layer_output['user']], dim=1)
#         combined_output['recipe'] = torch.cat([h['recipe'], first_layer_output['recipe']], dim=1)
#         combined_output['user'] = torch.add(h['user'], first_layer_output['user'])
#         combined_output['recipe'] = torch.add(h['recipe'], first_layer_output['recipe'])
#         return combined_output

###### LOSSES AND HELPER FUNCTIONS ######


def norm(input, p=1, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def get_recommendation_loss(pos_score, neg_score):
    n = pos_score.shape[0]
    return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()


def get_contrastive_loss(x1, x2, temperature):

    # users
    x1_user, x2_user = F.normalize(x1['user']), F.normalize(x2['user'])
    pos_score_user = torch.mul(x1_user, x2_user).sum(dim=1)
    pos_score_user = torch.exp(pos_score_user/temperature)

    x2_user_neg = torch.flipud(x2_user)
    ttl_score_user = torch.mul(x1_user, x2_user_neg).sum(dim=1)
    ttl_score_user = pos_score_user + torch.exp(ttl_score_user/temperature)

    contrastive_loss_user = - \
        torch.log(pos_score_user/ttl_score_user).mean()
    # print('contrastive_loss_user: ', contrastive_loss_user)
    assert not math.isnan(contrastive_loss_user)

    # recipes
    x1_recipe, x2_recipe = F.normalize(
        x1['recipe']), F.normalize(x2['recipe'])
    pos_score_recipe = torch.mul(x1_recipe, x2_recipe).sum(dim=1)
    pos_score_recipe = torch.exp(pos_score_recipe/temperature)

    x2_recipe_neg = torch.flipud(x2_recipe)
    ttl_score_recipe = torch.mul(x1_recipe, x2_recipe_neg).sum(dim=1)
    ttl_score_recipe = pos_score_recipe + \
        torch.exp(ttl_score_recipe/temperature)  # .sum(dim=1)

    contrastive_loss_recipe = - \
        torch.log(pos_score_recipe/ttl_score_recipe).mean()
    # print('contrastive_loss_recipe: ', contrastive_loss_recipe)

    return contrastive_loss_user + contrastive_loss_recipe


def get_emb_loss(*params):
    out = None
    for param in params:
        for k, v in param.items():
            if out == None:
                out = (v**2/2).mean()
            else:
                out += (v**2/2).mean()
    return out

###### MODEL ######

# Passo i parametri con le cose che mi servono li setto sotto self. che diventano attributi dell'istanza
# così sono accessibili da tutte le funzioni e classi nello scope dell'istanza


class RecipeRec(nn.Module):
    def __init__(self, device, ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, total_ingre_neighbor_tensor, graph, learning_rate, epochs, batch_size, drop_out, embedding_size, reg, temperature, attentions_heads, gamma):
        super().__init__()
        self.user_embedding = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
        )
        self.instr_embedding = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.ingredient_embedding = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU()
        )
        self.recipe_combine2out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.gnn = GNN(128, 128, 128, graph.etypes, attentions_heads)
        self.pred = ScorePredictor()
        self.setTransformer_ = SetTransformer(drop_out)
        self.ingre_neighbor_tensor = ingre_neighbor_tensor
        self.ingre_length_tensor = ingre_length_tensor
        self.total_length_index_list = total_length_index_list
        self.total_ingre_neighbor_tensor = total_ingre_neighbor_tensor
        self.graph = graph
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop_out = drop_out
        self.embedding_size = embedding_size
        self.reg = reg
        self.temperature = temperature
        self.attentions_heads = attentions_heads
        self.gamma = gamma

    # TODO probabilmente viene usata anche per fare evaluation e model.training è un flag che dice se sto facendo training oppure no.
    # devo settarlo a mano quando chiamo la funzione con True o False
    def forward(self, positive_graph, negative_graph, blocks, input_features, is_training=True):
        user, instr, ingredient, ingredient_of_dst_recipe = input_features

        # major GNN
        user_major = self.user_embedding(user)
        user_major = norm(user_major)
        instr_major = self.instr_embedding(instr)
        instr_major = norm(instr_major)
        ingredient_major = self.ingredient_embedding(ingredient)
        ingredient_major = norm(ingredient_major)
        x = self.gnn(blocks, {'user': user_major, 'recipe': instr_major,
                              'ingredient': ingredient_major}, torch.Tensor([[0]]))

        # contrastive - 1
        user1 = node_drop(user, self.drop_out[0], is_training)
        instr1 = node_drop(instr, self.drop_out[0], is_training)
        ingredient1 = node_drop(ingredient, self.drop_out[0], is_training)

        user1 = self.user_embedding(user1)
        user1 = norm(user1)
        instr1 = self.instr_embedding(instr1)
        instr1 = norm(instr1)
        ingredient1 = self.ingredient_embedding(ingredient1)
        ingredient1 = norm(ingredient1)

        x1 = self.gnn(blocks, {'user': user1, 'recipe': instr1,
                               'ingredient': ingredient1}, torch.Tensor([[1]]))

        # contrastive - 2
        user2 = node_drop(user, self.drop_out[0], is_training)
        instr2 = node_drop(instr, self.drop_out[0], is_training)
        ingredient2 = node_drop(ingredient, self.drop_out[0], is_training)

        user2 = self.user_embedding(user2)
        user2 = norm(user2)
        instr2 = self.instr_embedding(instr2)
        instr2 = norm(instr2)
        ingredient2 = self.ingredient_embedding(ingredient2)
        ingredient2 = norm(ingredient2)

        x2 = self.gnn(blocks, {'user': user2, 'recipe': instr2,
                               'ingredient': ingredient2}, torch.Tensor([[1]]))

        # setTransformer
        all_ingre_emb_for_each_recipe = get_ingredient_neighbors_all_embeddings(
            blocks, blocks[1].dstdata['_ID']['recipe'], ingredient_of_dst_recipe, self.device, self.ingre_neighbor_tensor, self.ingre_length_tensor)
        all_ingre_emb_for_each_recipe = norm(all_ingre_emb_for_each_recipe)
        total_ingre_emb = self.setTransformer_(
            all_ingre_emb_for_each_recipe)  # 1
        total_ingre_emb = norm(total_ingre_emb)

        # scores
        x['recipe'] = self.recipe_combine2out(
            total_ingre_emb.add(x['recipe']))
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)

        return pos_score, neg_score, x1, x2
