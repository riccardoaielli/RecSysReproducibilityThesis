#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2022

@author: Anonymized for blind review
"""

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO

import gc
import numpy as np
import torch
import scipy.sparse as sps

from SIGIR2022.GDE_our_interface.GDE_our_interface import GDE
from Utils.PyTorch.utils import get_optimizer

class GDE_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "GDE_RecommenderWrapper"

    def __init__(self, URM_train, verbose = True, use_gpu = False):
        super(GDE_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")

        self.temp_file_folder = None

        self.warm_user_ids = np.arange(0, self.n_users)[np.ediff1d(sps.csr_matrix(self.URM_train).indptr) > 0]
        # self.warm_user_ids = torch.from_numpy(self.warm_user_ids).to(self.device)


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # In order to compute the prediction the model may need a Session. The session is an attribute of this Wrapper.
        # There are two possible scenarios for the creation of the session: at the beginning of the fit function (training phase)
        # or at the end of the fit function (before loading the best model, testing phase)

        # predict_all = self.model.predict_matrix().detach().cpu().numpy()
        predict_batch = self._model.predict_batch(user_id_array).detach().cpu().numpy()

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf
            item_scores[:, items_to_compute] = predict_batch[:,items_to_compute]
        else:
            item_scores = predict_batch

        return item_scores

    def _init_model(self, spectral_features_dict = None):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        self._model = GDE(
            rating_matrix_sparse= self.URM_train,
            user_size=self.n_users,
            item_size=self.n_items,
            beta=self.beta,
            feature_type=self.feature_type,
            drop_out=self.drop_out,
            embedding_size=self.embedding_size,
            reg=self.reg,
            batch_size=self.batch_size,
            smooth_ratio = self.smooth_ratio,
            rough_ratio = self.rough_ratio,
            spectral_features_dict=spectral_features_dict,
            device = self.device,
        ).to(self.device)


    def fit(self,
            epochs = None,
            batch_size = None,
            learning_rate = None,
            beta = None,
            feature_type = None,
            drop_out = None,
            embedding_size = None,
            reg = None,
            smooth_ratio = None,
            rough_ratio = None,
            loss_type = None,

            # Additional Hyperparameters added by us
            # the default value produces the original setting
            sgd_mode = 'sgd',

            # These are standard
            **earlystopping_kwargs
            ):

        # set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.feature_type = feature_type
        self.drop_out = drop_out
        self.embedding_size = embedding_size
        self.reg = reg
        self.smooth_ratio = smooth_ratio
        self.rough_ratio = rough_ratio
        self.loss_type = loss_type

        self._print("Ensuring URM_train is implicit.")
        self.URM_train.data = np.ones_like(self.URM_train.data)

        self._init_model(spectral_features_dict = None)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, self.learning_rate, 0.0)

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

        gc.collect()
        torch.cuda.empty_cache()

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self._model.user_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(self._USER_factors_best).to(self.device))
        self._model.item_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(self._ITEM_factors_best).to(self.device))

        self._print("Training complete")


    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self._USER_factors_best = self._model.user_embed.weight.clone().detach().cpu().numpy()
        self._ITEM_factors_best = self._model.item_embed.weight.clone().detach().cpu().numpy()

    def _run_epoch(self, currentEpoch):
        total_loss = 0.0
        num_batches_per_epoch = int(self.URM_train.nnz / self.batch_size)
        # URM_train_tensor = _sps_to_tensor(self.URM_train).to(self.device)

        for j in range(0, num_batches_per_epoch):
            
            # Clear previously computed gradients
            self._optimizer.zero_grad()
            
            u = torch.LongTensor(np.random.choice(self.warm_user_ids, size=self.batch_size))
            # u = self.warm_user_ids[torch.randint(len(self.warm_user_ids), size=(self.batch_size,))]

            # Transferring only the sparse structure to reduce the data transfer
            user_batch_tensor = self.URM_train[u]
            user_batch_tensor = torch.sparse_csr_tensor(user_batch_tensor.indptr,
                                                        user_batch_tensor.indices,
                                                        user_batch_tensor.data,
                                                        size=user_batch_tensor.shape, dtype=torch.float32, device=self.device, requires_grad=False).to_dense()

            # ser_batch_tensor = _sps_to_tensor(self.URM_train[u], self.device).to_dense()
            # user_batch_tensor = torch.Tensor(self.URM_train[u].toarray()).to(self.device)

            p = torch.multinomial(user_batch_tensor, 1, True).squeeze(1)
            nega = torch.multinomial(1 - user_batch_tensor, 1, True).squeeze(1)

            loss = self._model(u, p, nega, self.loss_type)
            
            # Compute gradients given current loss
            loss.backward()
            total_loss += loss.item()

            # assert not np.isnan(loss.item()), "Loss is NAN."
            
            # Apply gradient using the selected _optimizer
            self._optimizer.step()



    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        spectral_features_dict = self._model.get_spectral_features()

        _USER_factors = self._model.user_embed.weight.clone().detach().cpu().numpy()
        _ITEM_factors = self._model.item_embed.weight.clone().detach().cpu().numpy()

        data_dict_to_save = {
            # Hyperparameters
            'beta': self.beta,
            'feature_type': self.feature_type,
            'drop_out': self.drop_out,
            'embedding_size': self.embedding_size,
            'reg': self.reg,
            'batch_size': self.batch_size,
            'smooth_ratio': self.smooth_ratio,
            'rough_ratio': self.rough_ratio,

            # Model parameters
            "model_spectral_features_dict": spectral_features_dict,
            "model_USER_factors": _USER_factors,
            "model_ITEM_factors": _ITEM_factors,
        }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            if not attrib_name.startswith("model_"):
                self.__setattr__(attrib_name, data_dict[attrib_name])

        self._init_model(spectral_features_dict = data_dict["model_spectral_features_dict"])

        _USER_factors = data_dict["model_USER_factors"]
        _ITEM_factors = data_dict["model_ITEM_factors"]

        self._model.user_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(_USER_factors).to(self.device))
        self._model.item_embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(_ITEM_factors).to(self.device))

        self._print("Loading complete")
