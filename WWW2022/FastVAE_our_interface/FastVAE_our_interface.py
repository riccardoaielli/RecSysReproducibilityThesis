# The cluster algorithmn(K-means) is implemented on the GPU
from operator import imod, neg
from numpy.core.numeric import indices
import scipy.sparse as ss
from sklearn import cluster
from sklearn.cluster import KMeans
import torch
import numpy as np
import torch.nn as nn
from torch._C import device, dtype
from typing import List


import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class BaseVAE(nn.Module):
    def __init__(self, num_item, dims, active='relu', dropout=0.5):
        """
        dims is a list for latent dims
        """
        super(BaseVAE, self).__init__()
        self.num_item = num_item

        self.dims = dims
        assert len(dims) == 2, 'Not supported dims'
        self.encode_layer_0 = nn.Embedding(
            self.num_item + 1, dims[0], padding_idx=0)
        self.encode_layer_1 = nn.Linear(dims[0], dims[1] * 2)

        self.decode_layer_0 = nn.Linear(dims[1], dims[0])
        # self._Item_Embeddings = nn.Embedding(self.num_item + 1, dims[0], padding_idx=0)
        self._Item_Embeddings = nn.Linear(dims[0], self.num_item + 1)

        self.dropout = nn.Dropout(dropout)

        if active == 'relu':
            self.act = F.relu
        elif active == 'tanh':
            self.act == F.tanh
        elif active == 'sigmoid':
            self.act == F.sigmoid
        else:
            raise ValueError('Not supported active function')

    def encode(self, item_id):
        # item_id is padded
        count_nonzero = item_id.count_nonzero(
            dim=1).unsqueeze(-1)  # batch_user * 1
        user_embs = self.encode_layer_0(item_id)  # batch_user * dims
        user_embs = torch.sum(user_embs, dim=1) / count_nonzero.pow(0.5)
        user_embs = self.dropout(user_embs)

        h = self.act(user_embs)
        h = self.encode_layer_1(h)
        mu, logvar = h[:, :self.dims[1]], h[:, self.dims[1]:]
        return mu, logvar

    def decode(self, user_emb_encode, items):
        user_emb = self.decode_layer_0(user_emb_encode)
        # item_embs = self._Item_Embeddings(items)
        # item_embs = F.normalize(item_embs)
        return self._Item_Embeddings(user_emb)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, pos_items, sampler=None):
        mu, logvar = self.encode(pos_items)
        z = self.reparameterize(mu, logvar)

        items = torch.arange(self.num_item + 1, device=z.device)
        self.pos_items = pos_items
        part_rats = self.decode(z, items)
        t00 = time.time()
        loss = self.loss_function(part_rats)
        t11 = time.time()
        return mu, logvar, loss, 0.0, t11-t00

    def kl_loss(self, mu, log_var, anneal=1.0, reduction=False):
        if reduction is True:
            return -anneal * 0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        else:
            return -anneal * 0.5 * torch.sum(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

    def loss_function(self, part_rats, prob_neg=None, pos_rats=None, prob_pos=None, reduction=False):
        # max_v, _ = torch.max(part_rats, dim=-1)
        # chuli = part_rats - max_v.unsqueeze(-1)
        # logits =  chuli  - torch.log(torch.sum(torch.exp(chuli), dim=-1)).unsqueeze(-1)

        logits = F.log_softmax(part_rats, dim=-1)
        idx_mtx = (self.pos_items > 0).double()
        if reduction is True:
            return -torch.sum(torch.gather(logits, 1, self.pos_items) * idx_mtx, dim=-1).mean()
        else:
            return -torch.sum(torch.gather(logits, 1, self.pos_items) * idx_mtx, dim=-1).sum()

    def _get_user_emb(self, user_his):
        user_emb, _ = self.encode(user_his)
        return self.decode_layer_0(user_emb)

    def _get_item_emb(self):
        return self._Item_Embeddings.weight[1:]


class VAE_Sampler(BaseVAE):
    def __init__(self, num_item, dims, active='relu', dropout=0.5):
        super(VAE_Sampler, self).__init__(
            num_item, dims, active=active, dropout=dropout)
        self._Item_Embeddings = nn.Embedding(
            self.num_item + 1, dims[0], padding_idx=0)

    def decode(self, user_emb_encode, items):
        user_emb = self.decode_layer_0(user_emb_encode)
        item_embs = self._Item_Embeddings(items)
        # return torch.matmul(user_emb.view(user_emb.shape[0], 1, -1), item_embs.transpose(1,2)).squeeze(1)
        # return user_emb.unsqueeze(1).bmm(item_embs.transpose(1,2)).squeeze(1)
        return (user_emb.unsqueeze(1) * item_embs).sum(-1)
        # return (user_emb.view(user_emb.shape[0], 1, -1) * item_embs).sum(-1)
        # return torch.einsum('ijk,ik->ij', item_embs, user_emb)

    def forward(self, pos_items, sampler):

        mu, logvar = self.encode(pos_items)
        z = self.reparameterize(mu, logvar)

        user_emb = self.decode_layer_0(z)
        with torch.no_grad():
            t0 = time.time()
            pos_prob, neg_items, neg_prob = sampler(user_emb, pos_items)
            t1 = time.time()
        pos_items_emb = self._Item_Embeddings(pos_items)
        neg_items_emb = self._Item_Embeddings(neg_items)

        pos_rat = (user_emb.unsqueeze(1) * pos_items_emb).sum(-1)
        neg_rat = (user_emb.unsqueeze(1) * neg_items_emb).sum(-1)
        t00 = time.time()
        loss = self.loss_function(neg_rat, neg_prob, pos_rat, pos_prob)
        t11 = time.time()
        return mu, logvar, loss, t1 - t0, t11 - t00

    def loss_function(self, part_rats, log_prob_neg=None, pos_rats=None, log_prob_pos=None, reduction=False):
        idx_mtx = (pos_rats != 0).double()
        new_pos = pos_rats - log_prob_pos.detach()
        new_neg = part_rats - log_prob_neg.detach()

        # parts_log_sum_exp = torch.logsumexp(new_neg, dim=-1).unsqueeze(-1)
        # final = torch.log( torch.exp(new_pos) + torch.exp(parts_log_sum_exp))
        parts_sum_exp = torch.sum(torch.exp(new_neg), dim=-1).unsqueeze(-1)
        final = torch.log(torch.exp(new_pos) + parts_sum_exp)

        if reduction is True:
            return torch.sum((- new_pos + final) * idx_mtx, dim=-1).mean()
        else:
            return torch.sum((- new_pos + final) * idx_mtx, dim=-1).sum()

        # idx_mtx = (pos_rats != 0).double()
        # new_pos = pos_rats - log_prob_pos.detach()
        # new_neg = part_rats - log_prob_neg.detach()
        # # new_pos[pos_rats==0] = -np.inf
        # logits = torch.log_softmax(torch.cat([new_pos, new_neg], dim=-1), dim=-1)

        # num_pos_item = pos_rats.shape[1]

        # if reduction is True:
        #     return -torch.sum( logits[:, :num_pos_item] * idx_mtx, dim=-1).mean()
        # else:
        #     return -torch.sum( logits * idx_mtx, dim=-1).sum()


########################## sampler_gpu_mm ################################

def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int) is True:
        K = K_or_center
        C = X[torch.randperm(N, device=X.device)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * \
            (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = torch.zeros(N, K, device=X.device)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign, :])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C = (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count < .5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N, device=X.device)[:ndead]]
    return C, assign, assign_m, loss


def construct_index(cd01, K):
    # Stable is availabel in PyTorch 1.9. Earlier version is not supported.
    cd01, indices = torch.sort(cd01, stable=True)
    # save the indices according to the cluster
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K**2 + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr


class SamplerBase(nn.Module):
    """
        Uniformly Sample negative items for each query. 
    """

    def __init__(self, num_items, num_neg, device, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.num_neg = num_neg
        self.device = device

    def forward(self, query, pos_items=None, padding=0):
        """
        Input
            query: torch.tensor
                Sequential models:
                query: (B,L,D), pos_items : (B, L)
                Normal models:
                query: (B,D), pos_items: (B,L)
        Output
            pos_prob(None if no pos_items), neg_items, neg_prob
            pos_items.shape == pos_prob.shape
            neg_items.shape == neg_prob.shape
            Sequential models:
            neg_items: (B,L,N)
            Normal
        """
        assert padding == 0
        # for sequential models the number of queries is the B x L
        num_queries = np.prod(query.shape[:-1])
        neg_items = torch.randint(
            1, self.num_items + 1, size=(num_queries, self.num_neg), device=self.device)
        neg_items = neg_items.view(*query.shape[:-1], -1)
        neg_prob = -torch.log(self.num_items *
                              torch.ones_like(neg_items, dtype=torch.float))
        if pos_items is not None:
            pos_prob = -torch.log(self.num_items *
                                  torch.ones_like(pos_items, dtype=torch.float))
            return pos_prob, neg_items, neg_prob
        return None, neg_items, neg_prob


class PopularSampler(SamplerBase):
    def __init__(self, pop_count, num_neg, device, mode=0, **kwargs):
        super().__init__(pop_count.shape[0], num_neg, device)
        pop_count = torch.from_numpy(pop_count).to(self.device)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count) + 1e-16
        elif mode == 2:
            pop_count = pop_count**0.75

        pop_count = torch.cat([torch.zeros(1, device=self.device), pop_count])
        self.pop_prob = pop_count / pop_count.sum()
        self.table = torch.cumsum(self.pop_prob, -1)
        self.pop_prob[0] = torch.ones(1, device=self.device)

    def forward(self, query, pos_items=None, padding=0):
        assert padding == 0
        num_queris = np.prod(query.shape[:-1])
        seeds = torch.rand(num_queris, self.num_neg, device=self.device)
        neg_items = torch.searchsorted(self.table, seeds)
        neg_items = neg_items.view(*query.shape[:-1], -1)
        neg_prob = torch.log(self.pop_prob[neg_items])
        if pos_items is not None:
            pos_prob = torch.log(self.pop_prob[pos_items])
            return pos_prob, neg_items, neg_prob
        return None, neg_items, neg_prob


class MidxUniform(SamplerBase):
    """
        Midx Sampler with Uniform Variant
    """

    def __init__(self, item_embs: torch.tensor, num_neg, device, num_cluster, item_pop: torch.tensor = None, **kwargs):
        super().__init__(item_embs.shape[0], num_neg, device)
        if isinstance(num_cluster, int) is True:
            self.K = num_cluster
        else:
            self.K = num_cluster.size(0)

        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(embs1, num_cluster)
        self.c1, cd1, cd1m, _ = kmeans(embs2, num_cluster)

        # for retreival probability, considering padding
        self.c0_ = torch.cat(
            [torch.zeros(1, self.c0.size(1), device=self.device), self.c0], dim=0)
        # for retreival probability, considering padding
        self.c1_ = torch.cat(
            [torch.zeros(1, self.c1.size(1), device=self.device), self.c1], dim=0)

        # for retreival probability, considering padding
        self.cd0 = torch.cat(
            [torch.tensor([-1]).to(self.device), cd0], dim=0) + 1
        # for retreival probability, considering padding
        self.cd1 = torch.cat(
            [torch.tensor([-1]).to(self.device), cd1], dim=0) + 1

        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K)

        if item_pop is None:
            self.wkk = cd0m.T @ cd1m
        else:
            self.wkk = cd0m.T @ (cd1m * item_pop.view(-1, 1))

    def forward(self, query, pos_items=None, padding=0):
        assert padding == 0
        q0, q1 = query.view(-1, query.size(-1)).chunk(2, dim=-1)
        r1 = q1 @ self.c1.T
        r1s = torch.softmax(r1, dim=-1)  # num_q x K1
        r0 = q0 @ self.c0.T
        r0s = torch.softmax(r0, dim=-1)  # num_q x K0
        s0 = (r1s @ self.wkk.T) * r0s  # num_q x K0 | wkk: K0 x K1
        k0 = torch.multinomial(
            s0, self.num_neg, replacement=True)  # num_q x neg
        p0 = torch.gather(r0, -1, k0)     # num_q * neg

        subwkk = self.wkk[k0, :]          # num_q x neg x K1
        s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
        k1 = torch.multinomial(s1.view(-1, s1.size(-1)),
                               1).squeeze(-1).view(*s1.shape[:-1])  # num_q x neg
        p1 = torch.gather(r1, -1, k1)  # num_q x neg
        k01 = k0 * self.K + k1  # num_q x neg
        p01 = p0 + p1
        neg_items, neg_prob = self.sample_item(k01, p01)
        if pos_items is not None:
            pos_prop = self.compute_item_p(query, pos_items)
            return pos_prop, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
        return None, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01):
        # num_q x neg, the number of items
        item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
        item_idx = torch.floor(item_cnt * torch.rand_like(item_cnt,
                               dtype=torch.float32, device=self.device)).long()  # num_q x neg
        neg_items = self.indices[item_idx + self.indptr[k01]] + 1
        neg_prob = p01
        return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        k0 = self.cd0[pos_items]  # B x L || B x L1
        k1 = self.cd1[pos_items]  # B x L || B x L1
        c0 = self.c0_[k0, :]  # B x L x D || B x L1 x D
        c1 = self.c1_[k1, :]  # B x L x D || B x L1 x D
        q0, q1 = query.chunk(2, dim=-1)  # B x L x D || B x D
        if query.dim() == pos_items.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) +
                 torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1)  # B x L1
        else:
            r = torch.sum(c0 * q0, dim=-1) + \
                torch.sum(c1 * q1, dim=-1)  # B x L
        return r


class MidxUniPop(MidxUniform):
    """
    Popularity sampling for the final items
    """

    def __init__(self, item_embs: np.ndarray, num_neg, device, num_cluster, pop_count, mode=1, **kwargs):
        if mode == 0:
            pop_count = np.log(pop_count + 1)
        elif mode == 1:
            pop_count = np.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        pop_count = torch.tensor(pop_count, dtype=torch.float32, device=device)
        super(MidxUniPop, self).__init__(
            item_embs, num_neg, device, num_cluster, pop_count)

        # this is similar, to avoid log 0 !!! in case of zero padding
        self.p = torch.cat(
            [torch.ones(1, device=self.device), pop_count], dim=0)
        self.cp = pop_count[self.indices]
        for c in range(self.K**2):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(-1)
                self.cp[start:end] = cumsum / cumsum[-1]

    def forward(self, query, pos_items=None, padding=0):
        return super().forward(query, pos_items=pos_items, padding=padding)

    def sample_item(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        # print(maxlen)
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=self.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            start, dtype=torch.float32, device=self.device).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]] + 1
        # neg_probs = self.p[item_idx + self.indptr[k01] + 1] # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[neg_items]
        return neg_items, p01 + torch.log(neg_probs)

    def compute_item_p(self, query, pos_items):
        r = super().compute_item_p(query, pos_items)
        p_r = self.p[pos_items]
        return r + torch.log(p_r)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    device = 'cuda'
    from dataloader import RecData
    from utils import setup_seed

    setup_seed(10)
    data = RecData('datasets', 'amazoni')
    train, test = data.get_data(0.8)
    dim = 200
    user_num, num_items = train.shape
    num_neg = 2000
    num_cluster = 32
    max_iter = 200
    # item_embs = np.random.randn(num_items, dim)
    item_embs = torch.randn(num_items, dim, device=device) * 0.1
    pop_count = np.squeeze(train.sum(axis=0).A)
    device = torch.device(device)
    # sampler0 = SamplerBase(num_items, num_neg, device)
    # sampler1 = PopularSampler(pop_count, num_neg, device)
    sampler2 = MidxUniform(item_embs, num_neg, device, num_cluster)
    sampler3 = MidxUniPop(item_embs, num_neg, device, num_cluster, pop_count)
    batch_size = 1
    query = torch.randn(batch_size, dim, device=device) * 0.1

    # pop_item = torch.randint(0, num_items+1, size=(batch_size))
    # sampler0(query, pop_item)
    # sampler1(query, pop_item)
    # sampler2(query, pop_item)
    # sampler3(query, pop_item)
    count_tensor = torch.zeros(num_items, dtype=torch.long, device=device)
    for i in range(max_iter):
        _, neg_items, _ = sampler2(query)
        # _, neg_items, _ = sampler3(query)
        ids, counts = torch.unique(neg_items - 1, return_counts=True)
        count_tensor[ids] += counts
        # print(count_tensor.max(), counts.max())
    count_t = count_tensor / count_tensor.sum(-1)

    exact_prob = torch.softmax(torch.matmul(
        query, item_embs.T), dim=-1).squeeze()

    # =========================================
    # Plot prob
    item_ids = pop_count.argsort()

    exact_prob = exact_prob.cpu().data.numpy()
    count_prob = count_t.cpu().data.numpy()
    import matplotlib.pyplot as plt
    plt.plot(exact_prob[item_ids].cumsum(), label='Softmax', linewidth=4.0)
    plt.plot(count_prob[item_ids].cumsum(), label='Midx_Uni')
    plt.legend()
    plt.savefig('amazoni_check.jpg')


##################### utils ######################


def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_max_length(x):
    return max(x, key=lambda x: x.shape[0]).shape[0]


def pad_sequence_int(seq):
    def _pad(_it, _max_len):
        return np.concatenate((_it + 1, np.zeros(_max_len - len(_it), dtype=np.int32)))
    return [_pad(it, get_max_length(seq)) for it in seq]


def custom_collate_(batch):
    return torch.LongTensor(pad_sequence_int(batch))


class Eval:
    @staticmethod
    def evaluate_item(train: ss.csr_matrix, test: ss.csr_matrix, user: np.ndarray, item: np.ndarray, topk: int = 50, cutoff: int = 50):
        train = train.tocsr()
        test = test.tocsr()
        idx = np.squeeze((test.sum(axis=1) > 0).A)
        train = train[idx, :]
        test = test[idx, :]
        user = user[idx, :]
        N = train.shape[1]
        cand_count = N - train.sum(axis=1)
        if topk < 0:
            mat_rank = Eval.predict(train, test, user, item)
        else:
            mat_rank = Eval.topk_search_(train, test, user, item, topk)
        return Eval.compute_item_metric(test, mat_rank, cand_count, cutoff)

    @staticmethod
    def compute_item_metric(test: ss.csr_matrix, mat_rank: ss.csr_matrix, cand_count: np.ndarray, cutoff: int = 200):
        rel_count = (test != 0).sum(axis=1)
        istopk = mat_rank.max() < test.shape[1] * 0.5
        recall, precision, map = Eval.compute_recall_precision(
            mat_rank, rel_count, cutoff)
        ndcg = Eval.compute_ndcg(test, mat_rank, cutoff)
        if not istopk:
            auc, mpr = Eval.compute_auc(mat_rank, rel_count, cand_count)
            return {'item_recall': recall, 'item_prec': precision, 'item_map': map, 'item_ndcg': ndcg, 'item_mpr': mpr, 'item_auc': auc}
        else:
            return {'item_recall': recall, 'item_prec': precision, 'item_map': map, 'item_ndcg': ndcg}

    @staticmethod
    def compute_ndcg(test, mat_rank, cutoff):
        M, _ = test.shape
        mat_rank_ = mat_rank.tocoo()
        user, item, rank = mat_rank_.row, mat_rank_.col, mat_rank_.data
        score = np.squeeze(test[(user, item)].A) / np.log2(rank + 2)
        dcg_score = ss.csr_matrix((score, (user, rank)), shape=test.shape)
        dcg = np.cumsum(dcg_score[:, :cutoff].todense(), axis=1)
        dcg = np.c_[dcg, dcg_score.sum(axis=1)]
        idcg = np.zeros((M, cutoff+1))
        for i in range(M):
            r = test.data[test.indptr[i]:test.indptr[i+1]]
            idcg_ = np.cumsum(-np.sort(-r) /
                              np.log2(np.array(range(len(r)))+2))
            if cutoff > len(r):
                idcg[i, :] = np.r_[idcg_, np.tile(idcg_[-1], cutoff+1-len(r))]
            else:
                idcg[i, :] = np.r_[idcg_[:cutoff], idcg_[-1]]
        ndcg = dcg / idcg
        ndcg = np.mean(ndcg, axis=0)
        return np.squeeze(ndcg.A)

    @staticmethod
    def compute_recall_precision(mat_rank, user_count, cutoff):
        user_count = user_count.A.T
        M, _ = mat_rank.shape
        mat_rank_ = mat_rank.tocoo()
        user, rank = mat_rank_.row, mat_rank_.data
        user_rank = ss.csr_matrix(
            (np.ones_like(user), (user, rank)), shape=mat_rank.shape)
        user_rank = user_rank[:, :cutoff].todense()
        user_count_inv = ss.diags(1/user_count, [0])
        cum = np.cumsum(user_rank, axis=1)
        recall = np.mean(user_count_inv * cum, axis=0)
        prec_cum = cum * ss.diags(1/np.array(range(1, cutoff+1)), 0)
        prec = np.mean(prec_cum, axis=0)
        div = np.minimum(np.tile(range(1, cutoff+1), (M, 1)),
                         np.tile(user_count.T, (1, cutoff)))
        map = np.mean(np.divide(np.cumsum(np.multiply(
            prec_cum, user_rank), axis=1), div), axis=0)
        return np.squeeze(recall.A), np.squeeze(prec.A), np.squeeze(map.A)

    @staticmethod
    def compute_auc(mat_rank, rel_count, cand_count):
        rel_count = rel_count.A
        cand_count = cand_count.A
        tmp = mat_rank.sum(axis=1)
        mpr = np.mean(tmp / cand_count / rel_count)
        auc_vec = rel_count * cand_count - tmp - \
            rel_count - rel_count * (rel_count - 1) / 2
        auc_vec = auc_vec / ((cand_count - rel_count) * rel_count)
        auc = np.mean(auc_vec)
        return auc, mpr

    @staticmethod
    def evaluate_item_with_code(train: ss.csr_matrix, test: ss.csr_matrix, user: np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray], topk=200, cutoff=200):
        train = train.tocsr()
        test = test.tocsr()
        # result1 = Eval.topk_search_with_code(train, user, item_code, item_center, topk)
        result = Eval.topk_search_with_code_fast(
            train, user, item_code, item_center, topk)
        return Eval.evaluate_topk(train, test, result, cutoff)
    # @staticmethod
    # def topk_search_approximate(train:ss.csr_matrix, user:np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray]):

    @staticmethod
    def topk_search_with_code_fast(train: ss.csr_matrix, user: np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray], topk=200):
        M, _ = train.shape
        traind = [train.indices[train.indptr[i]:train.indptr[i + 1]].tolist()
                  for i in range(M)]
        center = np.concatenate(item_center)
        # result = uts.topk_search_with_code(traind, user, item_code, center, topk)
        # return result.reshape([M, topk])
        return None

    @staticmethod
    def topk_search_with_code(train: ss.csr_matrix, user: np.ndarray, item_code: np.ndarray, item_center: List[np.ndarray], topk=200):
        item_center = np.stack(item_center, 0)  # m x K x D
        M = train.shape[0]
        result = np.zeros((M, topk), dtype=np.int)
        for i in range(M):
            E = train.indices[train.indptr[i]:train.indptr[i + 1]]
            center_score = np.tensordot(
                item_center, user[i, :], [-1, -1])  # m x K
            # pred = uts.fetch_score(item_code, center_score)
            # pred[E] = -np.inf
            # idx = np.argpartition(pred, -topk)[-topk:]
            # result[i, :] = idx[np.argsort(-pred[idx])]
        return result

    @staticmethod
    def item_reranking(topk_item: np.ndarray, score_func):
        M, K = topk_item.shape
        result = np.zeros_like(topk_item)
        for i in range(M):
            score_item = [(topk_item[i, k], score_func(
                i, topk_item[i, k])) for k in range(K)]
            result[i, :] = [a for (a, b) in sorted(
                score_item, key=lambda x: -x[1])]
        return result

    @staticmethod
    def evaluate_topk(train: ss.csr_matrix, test: ss.csr_matrix, topk_item: np.ndarray, cutoff: int = 200):
        train = train.tocsr()
        test = test.tocsr()
        result = topk_item
        N = train.shape[1]
        cand_count = N - train.sum(axis=1)
        M = test.shape[0]
        uir = []
        for i in range(M):
            R = set(test.indices[test.indptr[i]:test.indptr[i+1]])
            for k in range(result.shape[1]):
                if result[i, k] in R:
                    uir.append((i, result[i, k], k))
        user_id, item_id, rank = zip(*uir)
        mat_rank = ss.csr_matrix((rank, (user_id, item_id)), shape=test.shape)
        return Eval.compute_item_metric(test, mat_rank, cand_count, cutoff)

    @staticmethod
    def topk_search(train: ss.csr_matrix, user: np.ndarray, item: np.ndarray, topk: int = 200) -> np.ndarray:
        train = train.tocsr()
        M, _ = train.shape
        item_t = item.T
        result = np.zeros((M, topk), dtype=np.int)
        for i in range(M):
            E = train.indices[train.indptr[i]:train.indptr[i+1]]
            pred = np.matmul(user[i, :], item_t)
            # pred = np.tensordot(user[i,:], item, [0,-1])
            pred[E] = -np.inf
            idx = np.argpartition(pred, -topk)[-topk:]
            result[i, :] = idx[np.argsort(-pred[idx])]
        return result

    @staticmethod
    def topk_search_(train: ss.csr_matrix, test: ss.csr_matrix, user: np.ndarray, item: np.ndarray, topk: int = 200) -> ss.csr_matrix:
        M, _ = train.shape
        # traind = [train.indices[train.indptr[i]:train.indptr[i + 1]].tolist() for i in range(M)]
        # result = uts.topk_search(traind, user, item, topk).reshape([M, topk])
        result = Eval.topk_search(train, user, item, topk)
        uir = []
        for i in range(M):
            R = set(test.indices[test.indptr[i]:test.indptr[i+1]])
            for k in range(topk):
                if result[i, k] in R:
                    uir.append((i, result[i, k], k))
        user_id, item_id, rank = zip(*uir)
        mat_rank = ss.csr_matrix((rank, (user_id, item_id)), shape=test.shape)
        return mat_rank
        # user_id, rank = result.nonzero()
        # item_id = result[(user_id, rank)]
        # mat_rank = sp.csr_matrix((rank, (user_id, item_id)), shape=test.shape)
        # return mat_rank.multiply(test !=0)

    @staticmethod
    def predict(train: ss.csr_matrix, test: ss.csr_matrix, user: np.ndarray, item: np.ndarray) -> ss.csr_matrix:
        M, _ = train.shape
        item_t = item.T
        full_rank = np.zeros_like(test.data)
        for i in range(M):
            E = train.indices[train.indptr[i]:train.indptr[i+1]]
            R = test.indices[test.indptr[i]:test.indptr[i+1]]
            U = user[i, :]
            pred = np.matmul(U, item_t)
            pred[E] = -np.inf
            idx = np.argsort(-pred)
            rank = np.zeros_like(idx)
            rank[idx] = range(len(idx))
            full_rank[test.indptr[i]:test.indptr[i+1]] = rank[R]
        mat_rank = ss.csr_matrix(
            (full_rank, test.indices, test.indptr), shape=test.shape)
        return mat_rank

    @staticmethod
    def format(metric: dict):
        list_str = []
        for k, v in metric.items():
            if 'ndcg' in k:
                m_str = '{0:11}:[{1}, {2:.4f}]'.format(k, ', '.join(
                    '{:.4f}'.format(e) for e in v[(10-1)::10]), v[-1])
            elif not isinstance(v, np.ndarray):
                m_str = '{0:11}:{1:.4f}'.format(k, v)
            else:
                m_str = '{0:11}:[{1}]'.format(k, ', '.join(
                    '{:.4f}'.format(e) for e in v[(10-1)::10]))
            list_str.append(m_str)
        return '\n'.join(list_str)
