#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


import torch
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO

from WWW2023.AutoCF_our_interface.AutoCF_our_interface import *
import torch.utils.data as dataloader

import numpy as np
import tensorflow as tf
import os
import shutil
import scipy.sparse as sps

# from Conferences.CIKM.ExampleAlgorithm_github.main import get_model


class AutoCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "AutoCF_RecommenderWrapper"

    def __init__(self, URM_train, trnMat, tstMat, valMat, batch, tst_batch, verbose=True, use_gpu=True):
        super(AutoCF_RecommenderWrapper, self).__init__(
            URM_train, verbose=verbose)

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                print("GPU is not available, using cpu")
                self.device = torch.device("cpu:0")
        else:
            print("GPU is not available, using cpu")
            self.device = torch.device("cpu:0")

        print('device: ', self.device)

        self.trnMat = trnMat
        self.tstMat = tstMat
        self.valMat = valMat

        self.n_user, self.n_item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        self.allOneAdj = self.makeAllOne(self.torchBiAdj)
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(
            trnData, batch_size=batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(
            tstData, batch_size=tst_batch, shuffle=False, num_workers=0)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # TODO if the model in the end is either a matrix factorization algorithm or an ItemKNN/UserKNN
        #  you can have this class inherit from BaseMatrixFactorization, BaseItemSimilarityMatrixRecommender
        #  or BaseUSerSimilarityMatrixRecommender
        #  in which case you do not have to re-implement this function, you only need to set the
        #  USER_factors, ITEM_factors (see PureSVD) or W_Sparse (see ItemKNN) data structures in the FIT function
        # In order to compute the prediction the model may need a Session. The session is an attribute of this Wrapper.
        # There are two possible scenarios for the creation of the session: at the beginning of the fit function (training phase)
        # or at the end of the fit function (before loading the best model, testing phase)

        # Do not modify this
        # Create the full data structure that will contain the item scores
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        if items_to_compute is not None:
            item_indices = items_to_compute
        else:
            item_indices = self._item_indices

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # TODO this predict function should be replaced by whatever code is needed to compute the prediction for a user

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its index as many times as the
            # number of items
            item_score_user = self.model.predict([self._user_ones_vector*user_id, item_indices],
                                                 batch_size=100, verbose=0)

            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.ravel()[
                    items_to_compute]
            else:
                item_scores[user_index, :] = item_score_user.ravel()

        return item_scores

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        tf.reset_default_graph()

        # TODO Instantiate the model
        # Always clear the default graph if using tehsorflow

        self.model = get_model(num_users=self.n_users,
                               num_items=self.n_items,
                               num_factors=num_factors,
                               params=self._params,
                               input_dim=input_dim,
                               dims=dimensions_vae,
                               n_z=num_factors,
                               activations=activations,
                               loss_type='cross-entropy',
                               lr=learning_rate_cvae,
                               random_seed=random_seed,
                               print_step=10,
                               verbose=False)

    def fit(self,
            # TODO replace those hyperparameters with the ones you need
            epochs=None,
            batch_size=None,
            learning_rate=None,
            temperature=None,
            attentions_heads=None,
            drop_out=None,
            embedding_size=None,
            reg=None,
            gamma=None,

            # These are standard
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(
            input_temp_file_folder=temp_file_folder)

        # TODO replace the following code with what needed to create an instance of the model.
        #  Preferably create an init_model function
        #  If you are using tensorflow before creating the model call tf.reset_default_graph()

        # The following code contains various operations needed by another wrapper

        self._params = Params()
        self._params.lambda_u = lambda_u
        self._params.lambda_v = lambda_v
        self._params.lambda_r = lambda_r
        self._params.a = a
        self._params.b = b
        self._params.M = M
        self._params.n_epochs = epochs

        # These are the train instances as a list of lists
        # The following code processed the URM into the data structure the model needs to train
        self._train_users = []

        self.URM_train = sps.csr_matrix(self.URM_train)

        for user_index in range(self.n_users):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index + 1]

            user_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_users.append(list(user_profile))

        self._train_items = []

        self.URM_train = sps.csc_matrix(self.URM_train)

        for user_index in range(self.n_items):

            start_pos = self.URM_train.indptr[user_index]
            end_pos = self.URM_train.indptr[user_index + 1]

            item_profile = self.URM_train.indices[start_pos:end_pos]
            self._train_items.append(list(item_profile))

        self.URM_train = sps.csr_matrix(self.URM_train)

        self._init_model()

        # TODO Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow
        self.sess.close()
        self.sess = tf.Session()

        ###############################################################################
        # This is a standard training with early stopping part, most likely you won't need to change it

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

    def _prepare_model_for_validation(self):
        # TODO Most likely you won't need to change this function
        pass

    def _update_best_model(self):
        # TODO Most likely you won't need to change this function
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):
        # TODO replace this with the train loop for one epoch of the model

        n = self.ICM_train.shape[0]

        # for epoch in range(self._params.n_epochs):
        num_iter = int(n / self._params.batch_size)
        # gen_loss = self.cdl_estimate(data_x, params.cdl_max_iter)
        gen_loss = self.model.cdl_estimate(self.ICM_train, num_iter)
        self.model.m_theta[:] = self.model.transform(self.ICM_train)
        likelihood = self.model.pmf_estimate(
            self._train_users, self._train_items, None, None, self._params)
        loss = -likelihood + 0.5 * gen_loss * n * self._params.lambda_r

        self.USER_factors = self.model.m_U.copy()
        self.ITEM_factors = self.model.m_V.copy()

        logging.info("[#epoch=%06d], loss=%.5f, neg_likelihood=%.5f, gen_loss=%.5f" % (
            currentEpoch, loss, -likelihood, gen_loss))

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        # TODO replace this with the Saver required by the model
        #  in this case the neural network will be saved with the _weights suffix, which is rather standard
        self.model.save_weights(
            folder_path + file_name + "_weights", overwrite=True)

        # TODO Alternativley you may save the tensorflow model with a session
        saver = tf.train.Saver()
        saver.save(self.sess, folder_path + file_name + "_session")

        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "n_users": self.n_users,
            "n_items": self.n_items,
            "mf_dim": self.mf_dim,
            "layers": self.layers,
            "reg_layers": self.reg_layers,
            "reg_mf": self.reg_mf,
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name,
                         data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        # TODO replace this with what required to re-instantiate the model and load its weights,
        #  Call the init_model function you created before
        self._init_model()
        self.model.load_weights(folder_path + file_name + "_weights")

        # TODO If you are using tensorflow, you may instantiate a new session here
        # TODO reset the default graph to "clean" the tensorflow state
        tf.reset_default_graph()
        saver = tf.train.Saver()
        saver.restore(self.sess, folder_path + file_name + "_session")

        self._print("Loading complete")
