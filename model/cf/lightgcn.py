'''
Reference:
    [1]Xiangnan He et al. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International
    ACM SIGIR conference on research and development in Information Retrieval, pages 639–648, 2020.
Reference:
    https://github.com/recsys-benchmark/DaisyRec-v2.0
'''
import torch
import torch.nn as nn

import numpy as np
import scipy.sparse as sp
from model.cf.AbstractRecommender import GeneralRecommender


class LightGCN(GeneralRecommender):
    def __init__(self, args):
        '''
        LightGCN Recommender Class

        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, embedding dimension
        num_layers : int, number of ego layers
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        '''
        super(LightGCN, self).__init__(args)

        self.epochs = args.epochs
        self.lr = args.lr
        self.reg_1 = args.reg_1
        self.reg_2 = args.reg_2
        self.topk = args.k
        self.device = args.device
        self.user_num = args.num_users
        self.item_num = args.num_items

        # get this matrix from utils.get_inter_matrix and add it in config
        self.interaction_matrix = args.inter_matrix

        self.factor_num = args.factor_num
        self.num_layers = args.block_num

        self.embed_user = nn.Embedding(self.user_num, self.factor_num)
        self.embed_item = nn.Embedding(self.item_num, self.factor_num)

        # self.predict_layer = nn.Linear(self.factor_num, 1)

        self.loss_type = args.loss_type
        self.optimizer = args.optimizer if args.optimizer != 'default' else 'adam'
        self.initializer = args.init_method if args.init_method != 'default' else 'xavier_normal'
        self.early_stop = args.early_stop

        # storage variables for rank evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(self._init_weight)

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

    def get_norm_adj_mat(self):
        '''
        Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            Sparse tensor of the normalized interaction matrix.
        '''
        # build adj matrix
        A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # norm adj matrix
        sum_arr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sum_arr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D

        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

        return SparseL

    def get_ego_embeddings(self):
        ''' Get the embedding of users and items and combine to an new embedding matrix '''
        user_embeddings = self.embed_user.weight
        item_embeddings = self.embed_item.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_embedding, item_embedding = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])

        return user_embedding, item_embedding #pred.view(-1)#

    def calc_loss(self, batch):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = batch[0].to(self.device).long()
        pos_item = batch[1].to(self.device).long()

        embed_user, embed_item = self.forward()

        u_embeddings = embed_user[user]
        pos_embeddings = embed_item[pos_item]
        pos_pred = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)

        u_ego_embeddings = self.embed_user(user)
        pos_ego_embeddings = self.embed_item(pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)
            # add regularization term
            loss += self.reg_1 * (u_ego_embeddings.norm(p=1) + pos_ego_embeddings.norm(p=1))
            loss += self.reg_2 * (u_ego_embeddings.norm() + pos_ego_embeddings.norm())

        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device).long()
            neg_embeddings = embed_item[neg_item]
            neg_pred = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            neg_ego_embeddings = self.embed_item(neg_item)

            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (
                        u_ego_embeddings.norm(p=1) + pos_ego_embeddings.norm(p=1) + neg_ego_embeddings.norm(p=1))
            loss += self.reg_2 * (u_ego_embeddings.norm() + pos_ego_embeddings.norm() + neg_ego_embeddings.norm())

        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        return loss

    def predict(self, u, i):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embedding = self.restore_user_e[u]
        i_embedding = self.restore_item_e[i]
        pred = torch.matmul(u_embedding, i_embedding.t())

        return pred.cpu().item()

    def rank(self, test_loader):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        rec_ids = torch.tensor([], device=self.device)
        self.eval()
        with torch.no_grad():
            for us in test_loader:
                us = us.to(self.device)
                rank_list = self.full_rank(us)

                rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy().astype(np.int32)

    def full_rank(self, u):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        user_emb = self.restore_user_e[u]  # factor
        items_emb = self.restore_item_e.data  # item * factor
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0))

        return torch.argsort(scores, descending=True)[:, :self.topk]
