"""
Module description:

"""

__version__ = ''
__author__ = ''
__email__ = ''

from abc import ABC

import torch
import numpy as np
import random


class SimpleXModel(torch.nn.Module, ABC):

    def __init__(self,
                 num_users: int,
                 num_items: int,
                 interaction_matrix: torch.Tensor,
                 factors: int,
                 learning_rate: float,
                 lw: float,
                 margin: float,
                 negative_weight: float,
                 g: float,
                 dprob: float,
                 random_seed: int,
                 name="SimpleX",
                 **kwargs
                 ):
        super().__init__()

        # set seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users  # number of users - |U|
        self.num_items = num_items  # number of items - |I|
        self.embed_k = factors  # embedding dimension  - k
        self.learning_rate = learning_rate  # learning rate
        self.l_w = lw  # L2 regularization weight - \lambda_r
        self.margin = margin
        self.negative_weight = negative_weight
        self.g = g
        self.dprob = dprob

        self.History = interaction_matrix  # non-trainable (num_users, num_items) -

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k) # ( num_users, k ) - user embeddings
        torch.nn.init.normal_(self.Gu.weight,std=1e-4)
        self.Gu.to(self.device)

        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)  # (num_items, k) - items embeddings
        torch.nn.init.normal_(self.Gi.weight,std=1e-4)
        self.Gi.to(self.device)

        # Matrix V - Eq. 5
        self.V = torch.nn.Linear(self.embed_k, self.embed_k, bias=False)
        torch.nn.init.normal_(self.V.weight,std=1e-4)
        self.V.to(self.device)

        # Dropout - not present in the original paper, but in the author released code
        self.dropout = torch.nn.Dropout(self.dprob)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def predict(self, start, stop, **kwargs) -> torch.Tensor:
        '''
        .to(self.device)
        start: first user_id in the batch
        stop: last user_id in the batch
        return: a torch.Tensor, shape (stop:star, |I|) where each entry [x, y] denotes the score for the user x w.r.t item y
        '''
        users_histories = self.History[start:stop].to(self.device) # retrieve users histories
        p_u = torch.matmul(users_histories, self.Gi.weight)  # Eq. (2)
        h_u = self.g * self.Gu.weight[start:stop] + (1 - self.g) * self.V(p_u)  # user_embedding h_u - Eq. (5)

        h_u = torch.nn.functional.normalize(h_u, dim=1)

        Gi = torch.nn.functional.normalize(self.Gi.weight.to(self.device), dim=1)
        # Compute score for each user in the batch wrt all the items
        return torch.matmul(h_u.to(self.device),
                            torch.transpose(Gi.to(self.device), 0, 1))


    def cosine_similarity(self, user_emb, item_emb):
        '''

        Args:
            user_emb: torch.Tensor [n_user_batch, embedding_size]
            item_emb: torch.Tensor [n_user_batch, num_item_batch (negative), embedding_size]

        Returns: Cosine Similarity between user and items, shape : [user_num, item_num]

        '''
        user_emb = torch.nn.functional.normalize(user_emb, dim=1)
        user_emb = user_emb.unsqueeze(2)
        item_emb = torch.nn.functional.normalize(item_emb, dim=2)
        user_item_cos_sim = torch.matmul(item_emb, user_emb)
        return user_item_cos_sim.squeeze(2)

    def ccl(self, pos_cos, neg_cos):
        pos_loss = torch.relu(1 - pos_cos)
        neg_loss = torch.relu(neg_cos - self.margin)
        neg_loss = neg_loss.mean(1, keepdim= True) * self.negative_weight
        ccl_loss = (pos_loss + neg_loss).mean()
        return ccl_loss

    def train_step(self, batch):
        user, pos, neg = batch

        # User representation
        # user_emb = self.Gu(user)
        users_histories = torch.squeeze(self.History[user.to('cpu')]).to(self.device)
        p_u = torch.matmul(users_histories, self.Gi.weight)  # Eq. (2)
        # p_u = self.dropout(p_u)

        h_u = self.g * self.Gu(user) + (1 - self.g) * self.V(p_u)  # user_embedding h_u - Eq. (5)
        h_u = self.dropout(h_u)
        # Positive and Negative items:
        e_pos = self.Gi(pos.to(self.device))   # positive item embedding
        e_neg = self.Gi(neg.to(self.device))   # negative items embeddings

        #  Cosine Similarity
        pos_cos = self.cosine_similarity(h_u, e_pos.unsqueeze(1))
        neg_cos = self.cosine_similarity(h_u, e_neg)

        ccl_loss = self.ccl(pos_cos, neg_cos)  # cosine contrastive loss
        # --- L2 Regularization ---
        reg_loss = self.l_w * (1 / 2) * (h_u.norm(2).pow(2) +
                                         e_pos.norm(2).pow(2) +
                                         e_neg.norm(2).pow(2)) / user.shape[0]

        loss = ccl_loss + reg_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)