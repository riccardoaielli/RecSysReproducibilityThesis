#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/02/2023

@author: Anonymized for blind review
"""

import torch, copy, math, collections, random
import numpy as np
from tqdm import tqdm


class KB_RelationsIterator(object):
    """
    This Sampler performs the following:
    - Uniform sampling of a positive head -> pos_head
    - Uniform sampling among the relations of the pos_head -> (pos_relation, pos_tail)

    - (1) Uniform sampling of a negative head -> neg_head
    - Uniform sampling among the relations of the neg_head, only the tail node is kept -> neg_tail
    - If the relation (pos_relation, neg_tail) is among the existing relations for pos_head, repeat from (1)

    The sample is: pos_head, pos_relation, pos_tail, neg_tail
    """

    def __init__(self, knowledge_base_df, batch_size = 1, set_n_samples_to_draw = None, verbose = True):
        super(KB_RelationsIterator, self).__init__()

        self.batch_size = batch_size
        self.batch_pos_head = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_pos_relation = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_pos_tail = torch.empty((self.batch_size,), dtype=torch.long)
        self.batch_neg_tail = torch.empty((self.batch_size,), dtype=torch.long)

        # Create dictionary structure used in the reproduced paper
        print("Initializing KB_RelationsIterator")

        self.kg_head_to_relation_tuple = collections.defaultdict(list)
        for index in tqdm(knowledge_base_df.index) if verbose else knowledge_base_df.index:
            relation_tuple = (knowledge_base_df.loc[index, "relation"], knowledge_base_df.loc[index, "tail"])
            self.kg_head_to_relation_tuple[knowledge_base_df.loc[index, "head"]].append(relation_tuple)

        self.kg_head_list = np.array(list(self.kg_head_to_relation_tuple.keys()))

        self.n_samples_available = len(self.kg_head_list)
        self.n_samples_to_draw = self.n_samples_available if set_n_samples_to_draw is None else set_n_samples_to_draw
        self.n_samples_drawn = 0


    def __len__(self):
        return math.ceil(self.n_samples_to_draw/self.batch_size)

    def __iter__(self):
        self.n_sampled_points = 0
        return self

    def __next__(self):

        if self.n_sampled_points >= self.n_samples_to_draw:
            raise StopIteration

        this_batch_size = min(self.batch_size, self.n_samples_to_draw-self.n_sampled_points)
        self.n_sampled_points += this_batch_size

        index_batch = np.random.randint(self.n_samples_available, size = this_batch_size)

        pos_head_batch = self.kg_head_list[index_batch]

        for i_batch in range(this_batch_size):

            pos_head = pos_head_batch[i_batch]
            pos_relation, pos_tail = random.choice(self.kg_head_to_relation_tuple[pos_head])

            neg_tail_selected = False

            while not neg_tail_selected:

                neg_head = random.choice(self.kg_head_list)
                neg_tail = random.choice(self.kg_head_to_relation_tuple[neg_head])[1]

                if (pos_relation, neg_tail) not in self.kg_head_to_relation_tuple[pos_head]:
                    neg_tail_selected = True


            self.batch_pos_head[i_batch] = pos_head
            self.batch_pos_relation[i_batch] = pos_relation
            self.batch_pos_tail[i_batch] = pos_tail
            self.batch_neg_tail[i_batch] = neg_tail

        return self.batch_pos_head[:i_batch+1], \
               self.batch_pos_relation[:i_batch+1], \
               self.batch_pos_tail[:i_batch + 1], \
               self.batch_neg_tail[:i_batch+1]


