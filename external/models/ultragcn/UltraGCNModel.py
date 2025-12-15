from abc import ABC

import torch
import numpy as np
import random


class UltraGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 w1,
                 w2,
                 w3,
                 w4,
                 initial_weight,
                 negative_num,
                 negative_weight,
                 ii_neighbor_mat,
                 ii_constraint_mat,
                 constraint_mat,
                 gamma,
                 lm,
                 random_seed,
                 name="UltraGCN",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        #  Weights for the components of the omega term in the loss function (related to user/item degrees). Floats.
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.initial_weight = initial_weight
        self.negative_num = negative_num
        self.negative_weight = negative_weight
        self.ii_neighbor_mat = ii_neighbor_mat  # Precomputed tensor mapping each item to its top N nearest neighbors (indices). Shape: (num_items, ii_num_neighbors
        self.ii_constraint_mat = ii_constraint_mat  # Precomputed tensor containing the similarity scores corresponding to ii_neighbor_mat. Shape: (num_items, ii_num_neighbors)
        self.constraint_mat = constraint_mat  # Dictionary containing precomputed 'beta' values based on user and item degrees (beta_uD, beta_iD). Tensors, Shape: (num_users,) and (num_items,) respectively.
        self.gamma = gamma  # L2 weight
        self.lm = lm  # Coefficient for the item-item collaborative loss L_I (lambda)

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.normal_(self.Gu.weight, std=self.initial_weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.normal_(self.Gi.weight, std=self.initial_weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_omegas(self, users, pos_items, neg_items):
        '''
        Computes the omega weights used in the main loss (cal_loss_l), based on the UltraGCN formulation involving
        user/item degrees. These weights adjust the importance of each positive and negative sample in the loss calculation.

        :param users:
        :param pos_items: Shape: (batch_size, )
        :param neg_items: Shape: (batch_size, negative_num).
        :return:
        '''
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(
                self.device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(self.device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                                   self.constraint_mat['beta_iD'][neg_items.flatten()]).to(self.device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(self.device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_l(self, users, pos_items, neg_items, omega_weight):
        '''
         Calculates the main BPR-like loss component (L_main in the paper), weighted by omega_weight. It encourages
         the score of positive items to be higher than negative items. Uses binary cross-entropy loss with logits
          for numerical stability.
        users: User IDs tensor. Shape: (batch_size,).
        pos_items: Positive item IDs tensor. Shape: (batch_size,).
        neg_items: Negative item IDs tensor. Shape: (batch_size, negative_num).
        omega_weight:  The weights calculated by get_omegas. Shape: (batch_size + batch_size * negative_num,).
        :return:
        '''
        user_embeds = self.Gu(users.to(self.device))  # (batch_size, embed_k).
        pos_embeds = self.Gi(pos_items.to(self.device)) # (batch_size, embed_k).
        neg_embeds = self.Gi(neg_items.to(self.device))  # (batch_size, negative_num, embed_k).

        # Compute dot product
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # (batch_size)
        user_embeds = user_embeds.unsqueeze(1)  # reshape for broadcasting (batch_size, 1, embed_k)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(self.device)  # (batch_size, negative_num)
        # Applies the negative part of the omega_weight, reshaped to match neg_scores
        #  Averages the loss across the negative samples for each user. Shape: (batch_size,).
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_scores, neg_labels,
                                                                        weight=omega_weight[len(pos_scores):].view(
                                                                            neg_scores.size()),
                                                                        reduction='none').mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(self.device)
        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_scores, pos_labels,
                                                                        weight=omega_weight[:len(pos_scores)],
                                                                        reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def cal_loss_i(self, users, pos_items):
        '''
        This loss encourages the user embedding to be similar to the embeddings of the neighbors of items
        they interacted with, weighted by the item-item similarity.
        :param users: (batch_size, )
        :param pos_items: (batch_size, )
        :return:
        '''
        # Looks up neighbors for each positive item, then gets their embeddings
        neighbor_embeds = self.Gi(
            self.ii_neighbor_mat[pos_items].to(self.device))  # (batch_size, ii_num_neighbors, embed_k)
        # Looks up precomputed similarities for these neighbors.
        sim_scores = self.ii_constraint_mat[pos_items].to(self.device)  # (batch_size, ii_num_neighbors)
        user_embeds = self.Gu(users.to(self.device)).unsqueeze(1)  # (batch_size, 1, embed_k)

        # dot product between user and neighbor embedding (fino a sum(dim=-1) )
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        # (batch_size, ii_num_neighbors)
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        # L2 regularization
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_l(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lm * self.cal_loss_i(users, pos_items)
        return loss

    def predict(self, users, **kwargs):
        items = torch.arange(self.num_items).to(self.device)
        user_embeds = self.Gu(users.to(self.device))
        item_embeds = self.Gi(items.to(self.device))

        # returned tensor (n_users_inbatch, num_items)
        return user_embeds.mm(item_embeds.t())

    def train_step(self, batch):
        user, pos, neg = batch
        loss = self.forward(user, pos, neg)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
