#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/02/2023

@author: Anonymized for blind review
"""


from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping

from Utils.PyTorch.utils import get_optimizer, _sps_to_coo_tensor, clone_pytorch_model_to_numpy_dict
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator
from normalized_adjacency_matrix import normalized_adjacency_matrix

from Recommenders.DataIO import DataIO
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch, copy

from SIGIR2022.KGCL_our_interface.KB_RelationsIterator import KB_RelationsIterator
from SIGIR2022.KGCL_our_interface.contrast import Contrast
from SIGIR2022.KGCL_github.code.GAT import GAT


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

def _from_KG_to_tensor_indices(knowledge_base_df, entities_per_head, n_items, n_entities, n_relations, device, verbose):
    """
    In the original implementation only 10 entities are selected for each head relation. There does not seem to be
    any specific ordering or criteria for their selection if not for which comes first in the data.

    :param knowledge_base_df:
    :param entities_per_head:
    :param n_items:
    :param n_entities:
    :param n_relations:
    :param device:
    :return:
    """

    tail_list = knowledge_base_df.groupby("head").agg({"tail": lambda x: list(x)[:entities_per_head]})
    relation_list = knowledge_base_df.groupby("head").agg({"relation": lambda x: list(x)[:entities_per_head]})

    # knowledge_base_df.groupby("head")['tail'].apply(list)
    # knowledge_base_df.groupby("head")['relation'].apply(list)

    i2es = dict()
    i2rs = dict()

    for item in tqdm(tail_list.index) if verbose else tail_list.index:
        new_tensor = torch.IntTensor(entities_per_head) + n_entities
        new_tensor[:len(tail_list.loc[item, "tail"])] = torch.IntTensor(tail_list.loc[item, "tail"])
        i2es[item] = new_tensor.to(device)

        new_tensor = torch.IntTensor(entities_per_head) + n_relations
        new_tensor[:len(relation_list.loc[item, "relation"])] = torch.IntTensor(relation_list.loc[item, "relation"])
        i2rs[item] = new_tensor.to(device)

    return i2es, i2rs


class KGCL(nn.Module):
    def __init__(self,
                 # config: dict,
                 # dataset: BasicDataset,
                 # kg_dataset
                 URM_train,
                 knowledge_base_df,
                 lightGCN_n_layers,
                 embedding_size,
                 dropout_rate,
                 entities_per_head,
                 device,
                 ):
        super(KGCL, self).__init__()
        # self.config = config
        # self.dataset: BasicDataset = dataset
        # self.kg_dataset = kg_dataset
        # self.__init_weight()
        # self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()

    # def __init_weight(self):

        self.num_users, self.num_items = URM_train.shape
        self.device = device
        self.num_entities = knowledge_base_df[["head", "tail"]].max().max() + 1
        self.num_relations = knowledge_base_df["relation"].max() + 1
        # print("user:{}, item:{}, entity:{}".format(self.num_users, self.num_items, self.num_entities))
        self.latent_dim = embedding_size
        self.n_layers = lightGCN_n_layers
        self.keep_prob = 1 - dropout_rate
        self.has_dropout = dropout_rate > 0.0
        self.A_split = False

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        # item and kg entity
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(num_embeddings=self.num_entities + 1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(num_embeddings=self.num_relations + 1, embedding_dim=self.latent_dim)

        # relation weights
        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        # world.cprint('use NORMAL distribution UI')
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        # world.cprint('use NORMAL distribution ENTITY')
        nn.init.normal_(self.embedding_entity.weight, std=0.1)
        nn.init.normal_(self.embedding_relation.weight, std=0.1)

        # if self.config['pretrain'] == 0:
        #     # world.cprint('use NORMAL distribution UI')
        #     nn.init.normal_(self.embedding_user.weight, std=0.1)
        #     nn.init.normal_(self.embedding_item.weight, std=0.1)
        #     # world.cprint('use NORMAL distribution ENTITY')
        #     nn.init.normal_(self.embedding_entity.weight, std=0.1)
        #     nn.init.normal_(self.embedding_relation.weight, std=0.1)
        # else:
        #     self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
        #     self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        #     print('use pretarined data')


        self.f = nn.Sigmoid()

        A_tilde = normalized_adjacency_matrix(URM_train, add_self_connection=False)
        self.Graph = _sps_to_coo_tensor(A_tilde, self.device).coalesce()

        self.kg_dict, self.item2relations = _from_KG_to_tensor_indices(knowledge_base_df, entities_per_head, self.num_items,
                                                                       self.num_entities, self.num_relations, device, False)

        # self.Graph = self.dataset.getSparseGraph()
        # self.ItemNet = self.kg_dataset.get_item_net_from_kg(self.num_items)
        # self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)

        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()
        # print(f"KGCL is ready to go!")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def view_computer_all(self, g_droped, kg_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(kg_droped)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.has_dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos, neg)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        # if (torch.isnan(loss).any().tolist()):
        #     print("user emb")
        #     print(userEmb0)
        #     print("pos_emb")
        #     print(posEmb0)
        #     print("neg_emb")
        #     print(negEmb0)
        #     print("neg_scores")
        #     print(neg_scores)
        #     print("pos_scores")
        #     print(pos_scores)
        #     return None
        return loss, reg_loss

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)  # (kg_batch_size, relation_dim)
        h_embed = self.embedding_item(h)  # (kg_batch_size, entity_dim)
        pos_t_embed = self.embedding_entity(pos_t)  # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)  # (kg_batch_size, entity_dim)
        # Equation (1)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)  # (kg_batch_size)
        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        # loss = kg_loss # + 1e-3 * l2_loss
        # loss = kg_loss
        return kg_loss, l2_loss

    # def calc_kg_loss(self, h, r, pos_t, neg_t):
    #     """
    #     h:      (kg_batch_size)
    #     r:      (kg_batch_size)
    #     pos_t:  (kg_batch_size)
    #     neg_t:  (kg_batch_size)
    #     """
    #     r_embed = self.embedding_relation(r)  # (kg_batch_size, relation_dim)
    #     W_r = self.W_R[r]  # (kg_batch_size, entity_dim, relation_dim)
    #
    #     h_embed = self.embedding_item(h)  # (kg_batch_size, entity_dim)
    #     pos_t_embed = self.embedding_entity(pos_t)  # (kg_batch_size, entity_dim)
    #     neg_t_embed = self.embedding_entity(neg_t)  # (kg_batch_size, entity_dim)
    #
    #     r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
    #     r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
    #     r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
    #
    #     # Equation (1)
    #     pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
    #     neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)
    #
    #     # Equation (2)
    #     kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
    #     kg_loss = torch.mean(kg_loss)
    #
    #     l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
    #     # # TODO: optimize L2 weight
    #     loss = kg_loss + 1e-3 * l2_loss
    #     # loss = kg_loss
    #     return loss

    def cal_item_embedding_gat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(self.device))  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        entity_embs = self.embedding_entity(item_entities)  # item_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat(item_embs, entity_embs, padding_mask)

    def cal_item_embedding_rgat(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(self.device))  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.embedding_entity(item_entities)  # item_num, entity_num_each, emb_dim
        relation_embs = self.embedding_relation(item_relations)  # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict

        # if (world.kgcn == "GAT"):
        #     return self.cal_item_embedding_gat(kg)
        # elif world.kgcn == "RGAT":
        #     return self.cal_item_embedding_rgat(kg)
        # elif (world.kgcn == "MEAN"):
        #     return self.cal_item_embedding_mean(kg)
        # elif (world.kgcn == "NO"):
        #     return self.embedding_item.weight

        return self.embedding_item.weight

    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(self.device))  # item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))  # item_num, entity_num_each
        entity_embs = self.embedding_entity(item_entities)  # item_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities != self.num_entities, torch.ones_like(item_entities),
                                   torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs + entity_embs_mean

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma




class KGCL_RecommenderWrapper(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):

    RECOMMENDER_NAME = "KGCL_RecommenderWrapper"

    def __init__(self, URM_train, knowledge_base_df, use_gpu = True, use_cython_sampler = True, verbose = True):
        super(KGCL_RecommenderWrapper, self).__init__(URM_train, verbose = verbose)

        self.knowledge_base_df = knowledge_base_df.copy()
        self._data_iterator_class = InteractionAndNegativeIterator_cython if use_cython_sampler else InteractionAndNegativeIterator
        self._loss_function = None

        if use_gpu:
            assert torch.cuda.is_available(), "GPU is requested but not available"
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu:0")


    def fit(self,
            epochs = None,
            GNN_layers_K = None,
            contrastive_loss_temperature_tau = None,
            batch_size = None,
            embedding_size = None,
            GNN_dropout_rate = None,
            knowledge_graph_dropout_rate = None,
            user_interaction_dropout_rate = None,
            mix_ratio = None,
            uicontrast = None,
            entities_per_head = None,
            l2_reg = None,
            self_supervised_loss_weight = None,
            sgd_mode = None,
            learning_rate = None,
            learning_rate_milestones = None,
            **earlystopping_kwargs):
        
        self.l2_reg = l2_reg
        self.self_supervised_loss_weight = self_supervised_loss_weight

        self._KB_data_iterator = KB_RelationsIterator(self.knowledge_base_df, batch_size = batch_size, verbose=self.verbose)
        self._BPR_data_iterator = self._data_iterator_class(self.URM_train, batch_size = batch_size)

        torch.cuda.empty_cache()
        
        self._model = KGCL(self.URM_train,
                           knowledge_base_df = self.knowledge_base_df,
                           lightGCN_n_layers = GNN_layers_K,
                           embedding_size = embedding_size,
                           dropout_rate = GNN_dropout_rate,
                           entities_per_head = entities_per_head,
                           device = self.device)

        self._print("Creating Contrast model")
        self._contrast_model = Contrast(self._model,
                                        URM_train = self.URM_train,
                                        tau = contrastive_loss_temperature_tau,
                                        device = self.device,
                                        knowledge_graph_dropout_rate= knowledge_graph_dropout_rate,
                                        user_interaction_dropout_rate = user_interaction_dropout_rate,
                                        mix_ratio = mix_ratio,
                                        uicontrast = uicontrast,
                                        kgc_enable = True,
                                        ).to(self.device)

        self._optimizer = get_optimizer(sgd_mode.lower(), self._model, learning_rate, 0.0)
        self._learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones = learning_rate_milestones, gamma = 0.2)

        ###############################################################################
        ### This is a standard training with early stopping part

        # Initializing for epoch 0
        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name = self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()
        self._model_state = self._model_state_best


    def _prepare_model_for_validation(self):
        with torch.no_grad():
            self._model.eval()
            self.USER_factors, self.ITEM_factors = self._model.computer()

            # The original implementation applies a sigmoid on the dot product of the embeddings
            # But the sigmoid does not alter the relative ordering of the item preferences so for the top-k task
            # can be removed
            self.USER_factors = self.USER_factors.detach().cpu().numpy()
            self.ITEM_factors = self.ITEM_factors.detach().cpu().numpy()

            self._model_state = clone_pytorch_model_to_numpy_dict(self._model)
            self._model.train()


    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self._model_state_best = copy.deepcopy(self._model_state)


    def _run_epoch(self, num_epoch):

        self._model.train()

        KB_batch_iterator = tqdm(self._KB_data_iterator) if self.verbose else self._KB_data_iterator

        # STAGE 1: TransR training
        transr_loss = 0
        for batch in KB_batch_iterator:
            # Clear previously computed gradients
            self._optimizer.zero_grad()

            pos_head, pos_relation, pos_tail, neg_tail = batch
            pos_head = pos_head.to(self.device)
            pos_relation = pos_relation.to(self.device)
            pos_tail = pos_tail.to(self.device)
            neg_tail = neg_tail.to(self.device)

            # Compute the loss function of the current batch
            # l2reg 1e-3
            kg_loss, l2_loss = self._model.calc_kg_loss_transE(pos_head, pos_relation, pos_tail, neg_tail)
            loss = kg_loss + l2_loss*self.l2_reg

            # Compute gradients given current loss
            loss.backward()
            transr_loss += loss.item()

            # Apply gradient using the selected _optimizer
            self._optimizer.step()

        self._print("TransR Loss {:.2E}".format(transr_loss))



        # STAGE 2: Joint Learning with BPR
        BPR_batch_iterator = tqdm(self._BPR_data_iterator) if self.verbose else self._BPR_data_iterator

        # For SGL
        contrast_views = self._contrast_model.get_views()
        uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
        kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]

        epoch_loss = 0
        for batch in BPR_batch_iterator:
            # Clear previously computed gradients
            self._optimizer.zero_grad()

            user_batch, pos_item_batch, neg_item_batch = batch
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            # Compute the loss function of the current batch
            loss_BPR, loss_reg = self._model.bpr_loss(user_batch, pos_item_batch, neg_item_batch)

            # loss_ssl = list()
            usersv1_ro, itemsv1_ro = self._model.view_computer_all(uiv1, kgv1)
            usersv2_ro, itemsv2_ro = self._model.view_computer_all(uiv2, kgv2)

            # from SGL source
            items_uiv1 = itemsv1_ro[pos_item_batch]
            items_uiv2 = itemsv2_ro[pos_item_batch]
            l_item = self._contrast_model.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro)

            users_uiv1 = usersv1_ro[user_batch]
            users_uiv2 = usersv2_ro[user_batch]
            l_user = self._contrast_model.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro)

            # loss_ssl.extend([l_user * self.ssl_reg, l_item * self.ssl_reg])
            # l_ssl = torch.stack(loss_ssl).sum()

            l_ssl = l_item + l_user
            loss = loss_BPR + loss_reg*self.l2_reg + l_ssl * self.self_supervised_loss_weight

            # Compute gradients given current loss
            loss.backward()
            epoch_loss += loss.item()
    
            # Apply gradient using the selected _optimizer
            self._optimizer.step()


        # Apply decay of learning rate at the end of each epoch
        self._learning_rate_scheduler.step()

        self._print("Loss {:.2E}".format(epoch_loss))
        


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"USER_factors": self.USER_factors,
                             "ITEM_factors": self.ITEM_factors,
                             "_model_state": self._model_state,
                             }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)


        self._print("Saving complete")



