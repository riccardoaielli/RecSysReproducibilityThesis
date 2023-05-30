import torch
import numpy as np
import gc
import scipy.sparse as sps
import time

# Adj: adjacency matrix
# size: the number of required features
# largest: Ture (default) for k-largest (smoothed) and Flase for k-smallest (rough) eigenvalues
# niter: maximum number of iterations
def cal_spectral_feature(Adj, size, type='user', largest=True, niter=5, temp_file_folder = None):
    # params for the function lobpcg
    # k: the number of required features
    # largest: Ture (default) for k-largest (smoothed)  and Flase for k-smallest (rough) eigenvalues
    # niter: maximum number of iterations
    # for more information, see https://pytorch.org/docs/stable/generated/torch.lobpcg.html

    value, vector = torch.lobpcg(Adj, k=size, largest=largest, niter=niter)

    if largest == True:
        feature_file_name = 'smooth_' + type + '_features.npy'
        value_file_name = 'smooth_' + type + '_values.npy'

    else:
        feature_file_name = 'rough_' + type + '_features.npy'
        value_file_name = 'rough_' + type + '_values.npy'

    print('Computing ' + value_file_name + '...')
    np.save(temp_file_folder + value_file_name, value.cpu().numpy())
    print('Computing ' + feature_file_name + '...')
    np.save(temp_file_folder + feature_file_name, vector.cpu().numpy())


def preprocess(rate_matrix, smooth_ratio=0.1, rough_ratio=0.0, temp_file_folder = None):
    user_size, item_size = rate_matrix.shape

    # user degree and item degree
    D_u = rate_matrix.sum(1)
    D_i = rate_matrix.sum(0)

    # in the case any users or items have no interactions
    for i in range(user_size):
        if D_u[i] != 0:
            D_u[i] = 1 / D_u[i].sqrt()

    for i in range(item_size):
        if D_i[i] != 0:
            D_i[i] = 1 / D_i[i].sqrt()

    # (D_u)^{-0.5}*rate_matrix*(D_i)^{-0.5}
    rate_matrix = D_u.unsqueeze(1) * rate_matrix * D_i

    # clear GPU
    del D_u, D_i
    gc.collect()
    torch.cuda.empty_cache()

    # user-user matrix
    L_u = rate_matrix.mm(rate_matrix.t())

    # smoothed feautes for user-user relations
    cal_spectral_feature(L_u, int(smooth_ratio * user_size), type='user', largest=True, temp_file_folder = temp_file_folder)
    # rough feautes for user-user relations
    if rough_ratio != 0:
        cal_spectral_feature(L_u, int(rough_ratio * user_size), type='user', largest=False, temp_file_folder = temp_file_folder)

    # clear GPU
    del L_u
    gc.collect()
    torch.cuda.empty_cache()

    # item-item matrix
    print('Computing item-item matrix...')
    L_i = rate_matrix.t().mm(rate_matrix)

    # smoothed feautes for item-item relations
    cal_spectral_feature(L_i, int(smooth_ratio * item_size), type='item', largest=True, temp_file_folder = temp_file_folder)
    # rough feautes for item-item relations
    if rough_ratio != 0:
        cal_spectral_feature(L_i, int(rough_ratio * item_size), type='item', largest=False, temp_file_folder = temp_file_folder)




def cal_spectral_feature_return(Adj, size, type='user', largest=True, niter=5):
    # params for the function lobpcg
    # k: the number of required features
    # largest: Ture (default) for k-largest (smoothed)  and Flase for k-smallest (rough) eigenvalues
    # niter: maximum number of iterations
    # for more information, see https://pytorch.org/docs/stable/generated/torch.lobpcg.html

    value, vector = torch.lobpcg(Adj, k=size, largest=largest, niter=niter)

    if largest == True:
        feature_file_name = 'smooth_' + type + '_features'
        value_file_name = 'smooth_' + type + '_values'

    else:
        feature_file_name = 'rough_' + type + '_features'
        value_file_name = 'rough_' + type + '_values'

    print('Computing ' + value_file_name + '...')
    # np.save(value_file_name, value.cpu().numpy())
    print('Computing ' + feature_file_name + '...')
    # np.save(feature_file_name, vector.cpu().numpy())

    return {
            value_file_name: value.cpu().numpy(),
            feature_file_name: vector.cpu().numpy()
            }


def preprocess_return(rate_matrix, smooth_ratio=0.1, rough_ratio=0.0):
    user_size, item_size = rate_matrix.shape

    start_time = time.time()

    # user degree and item degree
    D_u = rate_matrix.sum(1)
    D_i = rate_matrix.sum(0)

    # in the case any users or items have no interactions
    for i in range(user_size):
        if D_u[i] != 0:
            D_u[i] = 1 / D_u[i].sqrt()

    for i in range(item_size):
        if D_i[i] != 0:
            D_i[i] = 1 / D_i[i].sqrt()

    # (D_u)^{-0.5}*rate_matrix*(D_i)^{-0.5}
    rate_matrix = D_u.unsqueeze(1) * rate_matrix * D_i

    # clear GPU
    del D_u, D_i
    gc.collect()
    torch.cuda.empty_cache()

    # user-user matrix
    L_u = rate_matrix.mm(rate_matrix.t())

    spectral_features_dict = {}

    # smoothed feautes for user-user relations
    spectral_features_dict.update(cal_spectral_feature_return(L_u, int(smooth_ratio * user_size), type='user', largest=True))
    # rough feautes for user-user relations
    if rough_ratio != 0:
        spectral_features_dict.update(cal_spectral_feature_return(L_u, int(rough_ratio * user_size), type='user', largest=False))

    # clear GPU
    del L_u
    gc.collect()
    torch.cuda.empty_cache()

    # item-item matrix
    print('Computing item-item matrix...')
    L_i = rate_matrix.t().mm(rate_matrix)

    # smoothed feautes for item-item relations
    spectral_features_dict.update(cal_spectral_feature_return(L_i, int(smooth_ratio * item_size), type='item', largest=True))
    # rough feautes for item-item relations
    if rough_ratio != 0:
        spectral_features_dict.update(cal_spectral_feature_return(L_i, int(rough_ratio * item_size), type='item', largest=False))

    print("Preprocessing done in {:.2f} sec".format(time.time()-start_time))

    return spectral_features_dict


def preprocess_return_sparse(URM_sparse, device, smooth_ratio=0.1, rough_ratio=0.0):
    user_size, item_size = URM_sparse.shape

    start_time = time.time()

    # user degree and item degree
    D_u = np.array(URM_sparse.sum(axis = 1)).squeeze()
    D_i = np.array(URM_sparse.sum(axis = 0)).squeeze()

    # in the case any users or items have no interactions
    D_u[D_u != 0] = 1/np.sqrt(D_u[D_u != 0])
    D_i[D_i != 0] = 1/np.sqrt(D_i[D_i != 0])

    # (D_u)^{-0.5}*rate_matrix*(D_i)^{-0.5}
    URM_sparse = sps.diags(D_u).dot(URM_sparse).dot(sps.diags(D_i))
    # rate_matrix = D_u.unsqueeze(1) * rate_matrix * D_i
    URM_sparse = sps.coo_matrix(URM_sparse)
    URM_sparse = torch.sparse_coo_tensor(np.vstack([URM_sparse.row, URM_sparse.col]), URM_sparse.data, URM_sparse.shape, device = device)


    # clear GPU
    del D_u, D_i
    gc.collect()
    torch.cuda.empty_cache()

    # user-user matrix
    L_u = URM_sparse.mm(URM_sparse.t())

    spectral_features_dict = {}

    # smoothed feautes for user-user relations
    spectral_features_dict.update(cal_spectral_feature_return(L_u, int(smooth_ratio * user_size), type='user', largest=True))
    # rough feautes for user-user relations
    if rough_ratio != 0:
        spectral_features_dict.update(cal_spectral_feature_return(L_u, int(rough_ratio * user_size), type='user', largest=False))

    # clear GPU
    del L_u
    gc.collect()
    torch.cuda.empty_cache()

    # item-item matrix
    print('Computing item-item matrix...')
    L_i = URM_sparse.t().mm(URM_sparse)

    # smoothed feautes for item-item relations
    spectral_features_dict.update(cal_spectral_feature_return(L_i, int(smooth_ratio * item_size), type='item', largest=True))
    # rough feautes for item-item relations
    if rough_ratio != 0:
        spectral_features_dict.update(cal_spectral_feature_return(L_i, int(rough_ratio * item_size), type='item', largest=False))

    print("Preprocessing done in {:.2f} sec".format(time.time()-start_time))

    return spectral_features_dict
