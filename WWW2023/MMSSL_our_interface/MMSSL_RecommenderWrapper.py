#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/12/18

@author: Maurizio Ferrari Dacrema
"""


import gc
import torch
import tqdm
import sys
import math
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.DataIO import DataIO
from Recommenders.BaseTempFolder import BaseTempFolder

import torch.utils.data as dataloader
import torch.optim as optim

import numpy as np
import tensorflow as tf
import os
import shutil
import scipy.sparse as sps
from WWW2023.MMSSL_our_interface.MMSSL.Models import *
# from WWW2023.MMSSL_our_interface.MMSSL_our_interface import *


class MMSSL_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):

    RECOMMENDER_NAME = "MMSSL_RecommenderWrapper"

    def __init__(self, URM_train, image_feats, text_feats, image_feat_dim, text_feat_dim, ui_graph, n_items, n_users, verbose=True, use_gpu=True):  # TODO
        super(MMSSL_RecommenderWrapper, self).__init__(
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

        self.config = dict()
        self.config['device'] = self.device
        self.config['n_users'] = n_users
        self.config['n_items'] = n_items

        # useless
        self.config['verbose'] = True
        self.config['lambda_coeff'] = 0.9
        self.config['early_stopping_patience'] = 7
        self.config['layers'] = 1
        self.config['mess_dropout'] = '[0.1, 0.1]'
        self.config['sparse'] = 1
        self.config['test_flag'] = 'part'
        self.config['metapath_threshold'] = 2
        self.config['sc'] = 1.0
        self.config['ssl_c_rate'] = 1.3
        self.config['ssl_s_rate'] = 0.8
        self.config['g_rate'] = 0.000029
        self.config['sample_num'] = 1
        self.config['sample_num_neg'] = 1
        self.config['sample_num_ii'] = 8
        self.config['sample_num_co'] = 2
        self.config['mask_rate'] = 0.75
        self.config['gss_rate'] = 0.85
        self.config['anchor_rate'] = 0.75
        self.config['feat_reg_decay'] = 1e-5
        self.config['ad1_rate'] = 0.2
        self.config['ad2_rate'] = 0.2
        self.config['ad_sampNum'] = 1
        self.config['ad_topk_multi_num'] = 100
        self.config['fake_gene_rate'] = 0.0001
        self.config['ID_layers'] = 1
        self.config['reward_rate'] = 1
        self.config['G_embed_size'] = 64
        self.config['model_num'] = 2
        self.config['negrate'] = 0.01
        self.config['cis'] = 25
        self.config['confidence'] = 0.5
        self.config['ii_it'] = 15
        self.config['isload'] = False
        self.config['isJustTest'] = False

        # train
        self.config['seed'] = 2022
        self.config['epoch'] = 1000
        self.config['embed_size'] = 64
        self.config['batch_size'] = 1024
        self.config['D_lr'] = 3e-4  # TODO togli hardcoded
        self.config['topk'] = 10
        self.config['cf_model'] = 'slmrec'
        self.config['cl_rate'] = 0.03
        self.config['norm_type'] = 'sym'
        self.config['Ks'] = '[10, 20, 50]'
        self.config['regs'] = '[1e-5,1e-5,1e-2]'
        self.config['lr'] = 0.00055
        self.config['emm'] = 1e-3
        self.config['L2_alpha'] = 1e-3
        self.config['weight_decay'] = 1e-4

        # GNN
        self.config['drop_rate'] = 0.2
        self.config['model_cat_rate'] = 0.55
        self.config['gnn_cat_rate'] = 0.55
        self.config['id_cat_rate'] = 0.36
        self.config['id_cat_rate1'] = 0.36
        self.config['head_num'] = 4
        self.config['dgl_nei_num'] = 8

        # GAN
        self.config['weight_size'] = '[64, 64]'
        self.config['G_rate'] = 0.0001
        self.config['G_drop1'] = 0.31
        self.config['G_drop2'] = 0.5
        self.config['gp_rate'] = 1
        self.config['real_data_tau'] = 0.005
        self.config['ui_pre_scale'] = 100

        # cl
        self.config['T'] = 1
        self.config['tau'] = 0.5
        self.config['geneGraph_rate'] = 0.1
        self.config['geneGraph_rate_pos'] = 2
        self.config['geneGraph_rate_neg'] = -1
        self.config['m_topk_rate'] = 0.0001
        self.config['log_log_scale'] = 0.00001

        self.image_feats = image_feats
        self.text_feats = text_feats
        self.image_feat_dim = image_feat_dim  # TODO forse posso mettere in config
        self.text_feat_dim = text_feat_dim  # TODO forse posso mettere in config
        self.ui_graph = self.ui_graph_raw = ui_graph

    def _compute_item_score(self, user_id_array, items_to_compute=None):  # TODO

        item_scores = - np.ones((len(user_id_array), self.n_items)) * np.inf

        print(user_id_array)

        allPred = t.mm(self.usrEmbeds[user_id_array], t.transpose(
            self.itmEmbeds, 1, 0)).detach().cpu().numpy()  # * (1 - trnMask) - trnMask * 1e8

        if items_to_compute is not None:
            item_scores[user_id_array,
                        items_to_compute] = allPred[user_id_array, items_to_compute]
        else:
            item_scores = allPred  # item_scores[user_id_array, :] = allPred

        if True:
            print(np.shape(item_scores))

        return item_scores

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        torch.cuda.empty_cache()

        self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(
            self.ui_graph_raw.todense()).to(self.config['device'])
        self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(
            self.ui_graph_raw.T.todense()).to(self.config['device'])
        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T
        self.ui_graph = self.matrix_to_tensor(
            self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(
            self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self._model = MMSSL(self.n_users,
                            self.n_items,
                            self.emb_dim,
                            self.weight_size,
                            self.mess_dropout,
                            self.image_feats,
                            self.text_feats,
                            self.config,  # TODO# TODO
                            ).to(self.config['device'])

    def fit(self,  # TODO
            emb_dim=None,
            weight_size=None,
            mess_dropout=None,
            lr=None,

            temp_file_folder=None,
            # These are standard
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(
            input_temp_file_folder=temp_file_folder)

        self.lr = lr
        self.emb_dim = emb_dim
        self.weight_size = weight_size
        self.mess_dropout = mess_dropout  # TODO

        # Inizializza il modello
        self._init_model()
        # Ottimizzatore
        self.D = Discriminator(self.n_items).to(self.config['device'])
        self.D.apply(self.weights_init)
        self.optim_D = optim.Adam(
            self.D.parameters(), lr=self.config['D_lr'], betas=(0.5, 0.9))

        self.optimizer_D = optim.AdamW(
            [{'params': self._model.parameters()},], lr=self.lr)

        def fac(epoch): return 0.96 ** (epoch / 50)
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optimizer_D, lr_lambda=fac)

        ###############################################################################
        # This is a standard training with early stopping part, most likely you won't need to change it

        gc.collect()
        torch.cuda.empty_cache()

        self._update_best_model()

        self._train_with_early_stopping(self.config['epoch'],
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        # serve per fare testing della compute item score
        # self.load_model("./result_experiments/__Temp_AutoCF_RecommenderWrapper_96339/", "_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

        self._prepare_model_for_validation()

    def _prepare_model_for_validation(self):

        pass

    def _update_best_model(self):
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):  # TODO

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        line_var_loss, line_g_loss, line_d_loss, line_cl_loss, line_var_recall, line_var_precision, line_var_ndcg = [], [], [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        contrastive_loss = 0.
        n_batch = self.n_train // self.config['batch_size'] + 1
        self.gene_u, self.gene_real, self.gene_fake = None, None, {}
        self.topk_p_dict, self.topk_id_dict = {}, {}

        for idx in tqdm(range(n_batch)):
            self._model.train()
            users, pos_items, neg_items = data_generator.sample()
            sample_time += time() - sample_t1

            with torch.no_grad():
                ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds, _, _, _, _, _, _ \
                    = self._model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
            ui_u_sim_detach = self.u_sim_calculation(
                users, ua_embeddings, ia_embeddings).detach()
            image_u_sim_detach = self.u_sim_calculation(
                users, image_user_embeds, image_item_embeds).detach()
            text_u_sim_detach = self.u_sim_calculation(
                users, text_user_embeds, text_item_embeds).detach()
            inputf = torch.cat(
                (image_u_sim_detach, text_u_sim_detach), dim=0)
            predf = (self.D(inputf))
            lossf = (predf.mean())
            u_ui = torch.tensor(self.ui_graph_raw[users].todense()).to(
                self.config['device'])
            u_ui = F.softmax(u_ui - self.config['log_log_scale']*torch.log(-torch.log(torch.empty(
                (u_ui.shape[0], u_ui.shape[1]), dtype=torch.float32).uniform_(0, 1).to(self.config['device'])+1e-8)+1e-8)/self.config['real_data_tau'], dim=1)  # 0.002
            u_ui += ui_u_sim_detach*self.config['ui_pre_scale']
            u_ui = F.normalize(u_ui, dim=1)
            inputr = torch.cat((u_ui, u_ui), dim=0)
            predr = (self.D(inputr))
            lossr = - (predr.mean())
            gp = self.gradient_penalty(self.D, inputr, inputf.detach())
            loss_D = lossr + lossf + self.config['gp_rate']*gp
            self.optim_D.zero_grad()
            loss_D.backward()
            self.optim_D.step()
            line_d_loss.append(loss_D.detach().data)

            G_ua_embeddings, G_ia_embeddings, G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds, G_user_emb, _, G_image_user_id, G_text_user_id, _, _ \
                = self._model(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)

            G_u_g_embeddings = G_ua_embeddings[users]
            G_pos_i_g_embeddings = G_ia_embeddings[pos_items]
            G_neg_i_g_embeddings = G_ia_embeddings[neg_items]
            G_batch_mf_loss, G_batch_emb_loss, G_batch_reg_loss = self.bpr_loss(
                G_u_g_embeddings, G_pos_i_g_embeddings, G_neg_i_g_embeddings)
            G_image_u_sim = self.u_sim_calculation(
                users, G_image_user_embeds, G_image_item_embeds)
            G_text_u_sim = self.u_sim_calculation(
                users, G_text_user_embeds, G_text_item_embeds)
            G_image_u_sim_detach = G_image_u_sim.detach()
            G_text_u_sim_detach = G_text_u_sim.detach()

            if idx % self.config['T'] == 0 and idx != 0:
                self.image_ui_graph_tmp = csr_matrix((torch.ones(len(self.image_ui_index['x'])), (
                    self.image_ui_index['x'], self.image_ui_index['y'])), shape=(self.n_users, self.n_items))
                self.text_ui_graph_tmp = csr_matrix((torch.ones(len(self.text_ui_index['x'])), (
                    self.text_ui_index['x'], self.text_ui_index['y'])), shape=(self.n_users, self.n_items))
                self.image_iu_graph_tmp = self.image_ui_graph_tmp.T
                self.text_iu_graph_tmp = self.text_ui_graph_tmp.T
                self.image_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                    self.csr_norm(self.image_ui_graph_tmp, mean_flag=True)
                ).to(self.config['device'])
                self.text_ui_graph = self.sparse_mx_to_torch_sparse_tensor(
                    self.csr_norm(self.text_ui_graph_tmp, mean_flag=True)
                ).to(self.config['device'])
                self.image_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                    self.csr_norm(self.image_iu_graph_tmp, mean_flag=True)
                ).to(self.config['device'])
                self.text_iu_graph = self.sparse_mx_to_torch_sparse_tensor(
                    self.csr_norm(self.text_iu_graph_tmp, mean_flag=True)
                ).to(self.config['device'])

                self.image_ui_index = {'x': [], 'y': []}
                self.text_ui_index = {'x': [], 'y': []}

            else:
                _, image_ui_id = torch.topk(G_image_u_sim_detach, int(
                    self.n_items*self.config['m_topk_rate']), dim=-1)
                self.image_ui_index['x'] += np.array(torch.tensor(users).repeat(
                    1, int(self.n_items*self.config['m_topk_rate'])).view(-1)).tolist()
                self.image_ui_index['y'] += np.array(
                    image_ui_id.cpu().view(-1)).tolist()
                _, text_ui_id = torch.topk(G_text_u_sim_detach, int(
                    self.n_items*self.config['m_topk_rate']), dim=-1)
                self.text_ui_index['x'] += np.array(torch.tensor(users).repeat(
                    1, int(self.n_items*self.config['m_topk_rate'])).view(-1)).tolist()
                self.text_ui_index['y'] += np.array(
                    text_ui_id.cpu().view(-1)).tolist()

            feat_emb_loss = self.feat_reg_loss_calculation(
                G_image_item_embeds, G_text_item_embeds, G_image_user_embeds, G_text_user_embeds)

            batch_contrastive_loss = 0
            batch_contrastive_loss1 = self.batched_contrastive_loss(
                G_image_user_id[users], G_user_emb[users])
            batch_contrastive_loss2 = self.batched_contrastive_loss(
                G_text_user_id[users], G_user_emb[users])

            batch_contrastive_loss = batch_contrastive_loss1 + batch_contrastive_loss2

            G_inputf = torch.cat((G_image_u_sim, G_text_u_sim), dim=0)
            G_predf = (self.D(G_inputf))

            G_lossf = -(G_predf.mean())
            batch_loss = G_batch_mf_loss + G_batch_emb_loss + G_batch_reg_loss + feat_emb_loss + \
                self.config['cl_rate']*batch_contrastive_loss + \
                self.config['G_rate']*G_lossf  # feat_emb_loss

            line_var_loss.append(batch_loss.detach().data)
            line_g_loss.append(G_lossf.detach().data)
            line_cl_loss.append(batch_contrastive_loss.detach().data)

            self.optimizer_D.zero_grad()
            batch_loss.backward(retain_graph=False)
            self.optimizer_D.step()

            loss += float(batch_loss)
            mf_loss += float(G_batch_mf_loss)
            emb_loss += float(G_batch_emb_loss)
            reg_loss += float(G_batch_reg_loss)

        del ua_embeddings, ia_embeddings, G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings

        if math.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        users_to_val = list(data_generator.val_set.keys())
        ret = self.test(users_to_val, is_val=True)
        training_time_list.append(t2 - t1)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'].data)
        pre_loger.append(ret['precision'].data)
        ndcg_loger.append(ret['ndcg'].data)
        hit_loger.append(ret['hit_ratio'].data)

        line_var_recall.append(ret['recall'][1])
        line_var_precision.append(ret['precision'][1])
        line_var_ndcg.append(ret['ndcg'][1])

        return

    def save_model(self, folder_path, file_name=None):  # TODO

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(
            folder_path + file_name))

        data_dict_to_save = {
            'model': self._model,
        }

        torch.save(data_dict_to_save, folder_path + file_name + '.mod')

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):  # TODO

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(
            folder_path + file_name))

        ckp = torch.load(folder_path + file_name + '.mod')
        self._model = ckp['model']
        self.opt = torch.optim.Adam(self._model.parameters(),
                                    lr=self.lr, weight_decay=0)

        self._print("Loading complete")

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
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.config['device'])

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
            self.config['device'])

        num_batches = (self.n_items - 1) // self.config['batch_size'] + 1
        indices = torch.arange(0, self.n_items).to(self.config['device'])
        u_sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * self.config['batch_size']:(i_b + 1) * self.config['batch_size']]
            sim = torch.mm(topk_u, item_final[index].T)
            sim_gt = torch.multiply(sim, (1-u_ui[:, index]))
            u_sim_list.append(sim_gt)

        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)
        return u_sim

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(
            np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        #
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).to(self.config['device'])

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1./2*(g_item_image**2).sum() + 1./2*(g_item_text**2).sum() \
            + 1./2*(g_user_image**2).sum() + 1./2*(g_user_text**2).sum()
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = self.config['feat_reg_decay'] * feat_reg
        return feat_emb_loss

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        def f(x): return torch.exp(x / self.config['tau'])

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())+1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()
