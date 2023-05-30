import torch
import torch.nn as nn
import numpy as np
import gc
from SIGIR2022.GDE_our_interface.preprocess_our_interface import preprocess_return, preprocess_return_sparse

# user_size, item_size: the number of users and items
# beta: The hyper-parameter of the weighting fucntion
# feature_type: (1) only use smoothed feautes (smoothed), (2) both smoothed and rough features (borh)
# drop_out: the ratio of drop out \in [0,1]
# latent_size: size of user/item embeddings
# reg: parameters controlling the regularization strength
class GDE(nn.Module):
    def __init__(self, rating_matrix_sparse, user_size, item_size, beta=5.0, feature_type='smoothed', drop_out=0.1, embedding_size=64, reg=0.01,
                 smooth_ratio = None, rough_ratio = None, batch_size=256, spectral_features_dict=None, device = None):
        super(GDE, self).__init__()

        # self.rating_matrix = rating_matrix
        self.device = device
        self.user_embed = torch.nn.Embedding(user_size, embedding_size)
        self.item_embed = torch.nn.Embedding(item_size, embedding_size)

        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

        self.beta = beta
        self.reg = reg
        self.drop_out = drop_out
        self.batch_size = batch_size
        if drop_out != 0:
            self.m = torch.nn.Dropout(drop_out)

        if spectral_features_dict is None:
            # spectral_features_dict = preprocess_return(self.rating_matrix, smooth_ratio, rough_ratio)
            self.spectral_features_dict = preprocess_return_sparse(rating_matrix_sparse, self.device, smooth_ratio, rough_ratio)
        else:
            self.spectral_features_dict = spectral_features_dict

        if feature_type == 'smoothed':
            user_filter = self.weight_feature(torch.Tensor(self.spectral_features_dict['smooth_user_values']).to(self.device))
            item_filter = self.weight_feature(torch.Tensor(self.spectral_features_dict['smooth_item_values']).to(self.device))

            user_vector = torch.Tensor(self.spectral_features_dict['smooth_user_features']).to(self.device)
            item_vector = torch.Tensor(self.spectral_features_dict['smooth_item_features']).to(self.device)


        elif feature_type == 'both':

            user_filter = torch.cat([self.weight_feature(torch.Tensor(self.spectral_features_dict['smooth_user_values']).to(self.device)) \
                                        , self.weight_feature(torch.Tensor(self.spectral_features_dict['rough_user_values']).to(self.device))])

            item_filter = torch.cat([self.weight_feature(torch.Tensor(self.spectral_features_dict['smooth_item_values']).to(self.device)) \
                                        , self.weight_feature(torch.Tensor(self.spectral_features_dict['rough_item_values']).to(self.device))])

            user_vector = torch.cat([torch.Tensor(self.spectral_features_dict['smooth_user_features']).to(self.device), \
                                     torch.Tensor(self.spectral_features_dict['rough_user_features']).to(self.device)], 1)

            item_vector = torch.cat([torch.Tensor(self.spectral_features_dict['smooth_item_features']).to(self.device), \
                                     torch.Tensor(self.spectral_features_dict['rough_item_features']).to(self.device)], 1)

        else:
            raise ValueError("Value for attribute 'feature_type' not recognized.")

        self.L_u = (user_vector * user_filter).mm(user_vector.t())
        self.L_i = (item_vector * item_filter).mm(item_vector.t())

        del user_vector, item_vector, user_filter, item_filter
        gc.collect()
        torch.cuda.empty_cache()

    def get_spectral_features(self):
        return self.spectral_features_dict.copy()

    def weight_feature(self, value):
        return torch.exp(self.beta * value)

    def forward(self, user, pos_item, nega_item, loss_type='adaptive'):

        if self.drop_out==0:
            final_user = self.L_u[user].mm(self.user_embed.weight)
            final_pos = self.L_i[pos_item].mm(self.item_embed.weight)
            final_nega= self.L_i[nega_item].mm(self.item_embed.weight)

        else:
            final_user = (self.m(self.L_u[user])*(1-self.drop_out)).mm(self.user_embed.weight)
            final_pos = (self.m(self.L_i[pos_item])*(1-self.drop_out)).mm(self.item_embed.weight)
            final_nega = (self.m(self.L_i[nega_item])*(1-self.drop_out)).mm(self.item_embed.weight)


        if loss_type=='adaptive':

            res_nega=(final_user*final_nega).sum(1)
            nega_weight=(1-(1-res_nega.sigmoid().clamp(max=0.99)).log10()).detach()
            out=((final_user*final_pos).sum(1)-nega_weight*res_nega).sigmoid()

        else:
            out=((final_user*final_pos).sum(1)-(final_user*final_nega).sum(1)).sigmoid()

        reg_term=self.reg*(final_user**2+final_pos**2+final_nega**2).sum()
        return (-torch.log(out).sum()+reg_term)/self.batch_size

    # def predict_matrix(self):
    #     final_user=self.L_u.mm(self.user_embed.weight)
    #     final_item=self.L_i.mm(self.item_embed.weight)
    #     #mask the observed interactions
    #     return (final_user.mm(final_item.t())).sigmoid()-self.rating_matrix*1000

    def predict_batch(self, user_id_array):
        final_user=self.L_u[user_id_array,:].mm(self.user_embed.weight)
        final_item=self.L_i.mm(self.item_embed.weight)
        #mask the observed interactions
        # return (final_user.mm(final_item.t())).sigmoid()-self.rating_matrix[user_id_array,:]*1000

        # Note: in the original source code the rating matrix was subtracted to the predicted scores weighted by 1000
        # this was done to ensure that already seen items have a low score. It is not necessary in our framework since the
        # previously seen items are removed later from the recommendation list.
        return (final_user.mm(final_item.t())).sigmoid()