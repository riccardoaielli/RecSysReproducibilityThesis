import datetime
import torch as t
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data as dataloader
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse as sp


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


def normalizeAdj(self, mat):
    degree = np.array(mat.sum(axis=-1))
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()


def makeTorchAdj(self, mat):
    # make ui adj
    a = sp.csr_matrix((self.n_user, self.n_user))
    b = sp.csr_matrix((self.n_item, self.n_item))
    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    mat = (mat + sp.eye(mat.shape[0])) * 1.0
    mat = self.normalizeAdj(mat)

    # make cuda tensor
    idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    shape = t.Size(mat.shape)
    return t.sparse.FloatTensor(idxs, vals, shape).to(t.device("cpu"))


def makeAllOne(self, torchAdj):
    idxs = torchAdj._indices()
    vals = t.ones_like(torchAdj._values())
    shape = torchAdj.shape
    return t.sparse.FloatTensor(idxs, vals, shape).to(t.device("cpu"))


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.n_item)
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
