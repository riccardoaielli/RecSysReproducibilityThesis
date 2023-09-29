import datetime
import torch as t
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data as dataloader
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse as sp
import torch as t
import numpy as np
import pickle
import os
from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F


# TimeLoggger
logmsg = ''
timemark = dict()
saveDefault = False


def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if save != None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)


if __name__ == '__main__':
    log('')

# Utils


def innerProduct(usrEmbeds, itmEmbeds):
    return t.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrast(nodes, allEmbeds, allEmbeds2=None):
    if allEmbeds2 is not None:
        pckEmbeds = allEmbeds[nodes]
        scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
    else:
        uniqNodes = t.unique(nodes)
        pckEmbeds = allEmbeds[uniqNodes]
        scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
    return scores

# DataHandler


def normalizeAdj(mat):
    degree = np.array(mat.sum(axis=-1))
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()


def makeTorchAdj(n_user, n_items, mat, device):
    # make ui adj
    a = sp.csr_matrix((n_user, n_user))
    b = sp.csr_matrix((n_items, n_items))
    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    mat = (mat + sp.eye(mat.shape[0])) * 1.0
    mat = normalizeAdj(mat)

    # make cuda tensor
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse.FloatTensor(idxs, vals, shape).to(device)


def makeAllOne(torchAdj, device):
    idxs = torchAdj._indices()
    vals = t.ones_like(torchAdj._values())
    shape = torchAdj.shape
    return t.sparse.FloatTensor(idxs, vals, shape).to(device)


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self, n_items):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(n_items)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])


# MODEL

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class AutoCF(nn.Module):
    def __init__(self, device, n_users, n_items, lr, epochs, latdim, reg, ssl_reg, decay, head, gcn_layer, gt_layer, seedNum, maskDepth, fixSteps, keepRate, eps):
        super(AutoCF, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(n_users, latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(n_items, latdim)))
        self.gcnLayers = nn.Sequential(
            *[GCNLayer() for i in range(gcn_layer)])
        self.gtLayers = nn.Sequential(
            *[GTLayer(latdim, head, device) for i in range(gt_layer)])
        self.n_users = n_users
        self.device = device

    def getEgoEmbeds(self):
        return t.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def forward(self, encoderAdj, decoderAdj=None):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoderAdj, embedsLst[-1])
            embedsLst.append(embeds)
        if decoderAdj is not None:
            for gt in self.gtLayers:
                embeds = gt(decoderAdj, embedsLst[-1])
                embedsLst.append(embeds)
        embeds = sum(embedsLst)
        return embeds[:self.n_users], embeds[self.n_users:]


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


class GTLayer(nn.Module):
    def __init__(self, latdim, head, device):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(t.empty(latdim, latdim)))
        self.kTrans = nn.Parameter(init(t.empty(latdim, latdim)))
        self.vTrans = nn.Parameter(init(t.empty(latdim, latdim)))
        self.head = head
        self.latdim = latdim
        self.device = device

    def forward(self, adj, embeds):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (
            rowEmbeds @ self.qTrans).view([-1, self.head, self.latdim // self.head])
        kEmbeds = (
            colEmbeds @ self.kTrans).view([-1, self.head, self.latdim // self.head])
        vEmbeds = (
            colEmbeds @ self.vTrans).view([-1, self.head, self.latdim // self.head])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], self.head]).to(self.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)  # eh

        resEmbeds = t.einsum('eh, ehd -> ehd', att,
                             vEmbeds).view([-1, self.latdim])
        tem = t.zeros([adj.shape[0], self.latdim]).to(self.device)
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds


class LocalGraph(nn.Module):
    def __init__(self, device, seedNum):
        super(LocalGraph, self).__init__()

        self.device = device
        self.seedNum = seedNum

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).to(self.device)
        noise = -t.log(-t.log(noise))
        return t.log(scores) + noise

    def forward(self, allOneAdj, embeds):
        # allOneAdj should be with self-loop
        # embeds should be zero-order embeds
        order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1))
        scores = self.makeNoise(scores)
        _, seeds = t.topk(scores, self.seedNum)
        return scores, seeds


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, device, n_users, maskDepth, n_items, keepRate):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.device = device
        self.n_users = n_users
        self.maskDepth = maskDepth
        self.n_items = n_items
        self.keepRate = keepRate

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(self.maskDepth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = t.logical_or(rowIdct, colIdct)

                if i != self.maskDepth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[t.logical_not(idct)]
                cols = cols[t.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = t.unique(t.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)
        sampNum = int((self.n_users + self.n_items) * self.keepRate)
        sampedNodes = t.randint(
            self.n_users + self.n_items, size=[sampNum]).to(self.device)
        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = t.unique(t.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (
                tem.shape[0] / (self.n_users + self.n_items)), tem.shape[0], (self.n_users + self.n_items))
        maskNodes.append(sampedNodes)
        maskNodes = t.unique(t.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (
                maskNodes.shape[0] / (self.n_users + self.n_items)), maskNodes.shape[0], (self.n_users + self.n_items))
            self.flag = True
            print('-----')

        encoderAdj = self.normalizeAdj(t.sparse.FloatTensor(
            t.stack([rows, cols], dim=0), t.ones_like(rows).to(self.device), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[t.randint(
            temNum, size=[adj._values().shape[0]]).to(self.device)]
        temCols = maskNodes[t.randint(
            temNum, size=[adj._values().shape[0]]).to(self.device)]

        newRows = t.concat(
            [temRows, temCols, t.arange(self.n_users+self.n_items).to(self.device), rows])
        newCols = t.concat(
            [temCols, temRows, t.arange(self.n_users+self.n_items).to(self.device), cols])

        # filter duplicated
        hashVal = newRows * (self.n_users + self.n_items) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (self.n_users + self.n_items)
        newRows = ((hashVal - newCols) / (self.n_users + self.n_items)).long()

        decoderAdj = t.sparse.FloatTensor(t.stack(
            [newRows, newCols], dim=0), t.ones_like(newRows).to(self.device).float(), adj.shape)
        return encoderAdj, decoderAdj
