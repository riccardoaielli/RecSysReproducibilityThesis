#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


import gc
import torch
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO
from Recommenders.BaseTempFolder import BaseTempFolder

from WWW2023.AutoCF_our_interface.AutoCF_our_interface import *
import torch.utils.data as dataloader

import numpy as np
import tensorflow as tf
import os
import shutil
import scipy.sparse as sps

# from Conferences.CIKM.ExampleAlgorithm_github.main import get_model


class AutoCF_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "AutoCF_RecommenderWrapper"

    def __init__(self, URM_train, trnMat, tstMat, valMat, batch, tstBat, verbose=True, use_gpu=True):
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

        self.n_users, self.n_items = trnMat.shape
        self.torchBiAdj = makeTorchAdj(
            self.n_users, self.n_items, self.trnMat, self.device)
        self.allOneAdj = makeAllOne(self.torchBiAdj, self.device)
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(
            trnData, batch_size=batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(
            tstData, batch_size=tstBat, shuffle=False, num_workers=0)

        # This is used in _compute_item_score
        self._item_indices = np.arange(0, self.n_items, dtype=np.int)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # TODO Fixare in modo personalizzato per il io algoritnmo
        # Do not modify this
        # Create the full data structure that will contain the item scores
        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        if items_to_compute is not None:
            item_indices = items_to_compute
        else:
            item_indices = self._item_indices

        usrEmbeds, itmEmbeds = self.model(
            self.handler.torchBiAdj, self.handler.torchBiAdj)

        allPreds = t.mm(usrEmbeds[user_id_array], t.transpose(
            itmEmbeds, 1, 0)).detach().cpu().numpy()  # * (1 - trnMask) - trnMask * 1e8

        if items_to_compute is not None:
            item_scores[user_index,
                        items_to_compute] = allPreds[items_to_compute]
        else:
            item_scores[user_index, :] = allPreds

        """ # TODO sistema predict per compute item score
            def predict(self, tstLoader, torchBiAdj, _model):
                for usr, trnMask in tstLoader:
                    usr = usr.long().to(self.device)  # usr è un tensore
                    trnMask = trnMask.to(self.device)
                    usrEmbeds, itmEmbeds = _model(
                        torchBiAdj, torchBiAdj)
                    # Penso generi tutte le prediction per un utente solo o forse per una batch
                    allPreds = t.mm(usrEmbeds[usr], t.transpose(
                        itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8

                return allPreds """

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # TODO this predict function should be replaced by whatever code is needed to compute the prediction for a user

            # The prediction requires a list of two arrays user_id, item_id of equal length
            # To compute the recommendations for a single user, we must provide its index as many times as the
            # number of items
            item_score_user = self._model.predict(
                self.tstLoader, self.torchBiAdj, self._model)

            # Do not modify this
            # Put the predictions in the correct items
            if items_to_compute is not None:
                item_scores[user_index, items_to_compute] = item_score_user.tensor.detach().numpy().ravel()[
                    items_to_compute]
            else:
                item_scores[user_index,
                            :] = item_score_user.tensor.detach().numpy().ravel()

        return item_scores

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        self._model = AutoCF(device=self.device,
                             n_users=self.n_users,
                             n_items=self.n_items,
                             lr=self.lr,
                             epochs=self.epochs,
                             batch=self.batch,
                             tstBat=self.tstBat,
                             latdim=self.latdim,
                             reg=self.reg,
                             ssl_reg=self.ssl_reg,
                             decay=self.decay,
                             head=self.head,
                             gcn_layer=self.gcn_layer,
                             gt_layer=self.gt_layer,
                             tstEpoch=self.tstEpoch,
                             seedNum=self.seedNum,
                             maskDepth=self.maskDepth,
                             fixSteps=self.fixSteps,
                             keepRate=self.keepRate,
                             eps=self.eps
                             ).to(self.device)

    def fit(self,
            # TODO rimuovi ciò che non è un iperparametro e passa come argomento
            epochs=None,
            batch=None,
            tstBat=None,
            lr=None,
            latdim=None,
            ssl_reg=None,
            decay=None,
            head=None,
            gcn_layer=None,
            gt_layer=None,
            tstEpoch=None,
            seedNum=None,
            reg=None,
            maskDepth=None,
            fixSteps=None,
            keepRate=None,
            eps=None,

            temp_file_folder=None,
            # These are standard
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(
            input_temp_file_folder=temp_file_folder)

        # TODO rimuovi ciò che non è un iperparametro e passa come argomento
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.tstBat = tstBat
        self.latdim = latdim
        self.reg = reg
        self.ssl_reg = ssl_reg
        self.decay = decay
        self.head = head
        self.gcn_layer = gcn_layer
        self.gt_layer = gt_layer
        self.tstEpoch = tstEpoch
        self.seedNum = seedNum
        self.maskDepth = maskDepth
        self.fixSteps = fixSteps
        self.keepRate = keepRate
        self.eps = eps

        # Inizializza il modello
        self._init_model()
        # Ottimizzatore
        self.opt = t.optim.Adam(self._model.parameters(),
                                lr=self.lr, weight_decay=0)
        self.masker = RandomMaskSubgraphs(
            self.device, self.n_users, self.maskDepth, self.n_items, keepRate)
        self.sampler = LocalGraph(self.device, self.seedNum)

        ###############################################################################
        # This is a standard training with early stopping part, most likely you won't need to change it

        gc.collect()  # TODO vanno lasciate?
        torch.cuda.empty_cache()

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        # self.load_model("./result_experiments/__Temp_AutoCF_RecommenderWrapper_96339/", "_best_model") serve per fare testing della compute item score

        # self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

    def _prepare_model_for_validation(self):
        # Most likely you won't need to change this function # TODO metti embeddings qui nell'oggetto che recupero nella compute item score
        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):

        trnLoader = self.trnLoader
        trnLoader.dataset.negSampling(self.n_items)
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // self.batch
        for i, tem in enumerate(trnLoader):
            if i % self.fixSteps == 0:
                sampScores, seeds = self.sampler(
                    self.allOneAdj, self._model.getEgoEmbeds())
                encoderAdj, decoderAdj = self.masker(
                    self.torchBiAdj, seeds)
            ancs, poss, _ = tem
            ancs = ancs.long().to(self.device)
            poss = poss.long().to(self.device)
            usrEmbeds, itmEmbeds = self._model(encoderAdj, decoderAdj)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]

            bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
            regLoss = calcRegLoss(self._model) * self.reg

            contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)
                            ) * self.ssl_reg + contrast(ancs, usrEmbeds, itmEmbeds)

            loss = bprLoss + regLoss + contrastLoss

            if i % self.fixSteps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self._print('Epoch: {}'.format(currentEpoch))
            log(' Step %d/%d: loss = %.1f, reg = %.1f, cl = %.1f   ' %
                (i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        data_dict_to_save = {
            'model': self._model,
        }

        t.save(data_dict_to_save, folder_path + file_name + '.mod')
        log('Model Saved: %s' % folder_path + file_name)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        ckp = t.load(folder_path + file_name + '.mod')
        self._model = ckp['model']
        self.opt = t.optim.Adam(self._model.parameters(),
                                lr=self.lr, weight_decay=0)

        self._print("Loading complete")
