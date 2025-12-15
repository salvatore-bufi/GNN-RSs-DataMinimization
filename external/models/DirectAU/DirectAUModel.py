"""
Module description: Implementation of the model proposed in "Towards Representation Alignment and Uniformity in Collaborative Filtering"

"""

__version__ = '0.0.1'
__author__ = 'Salvatore Bufi'
__email__ = 's.bufi@phd.poliba.it'

from abc import ABC

import torch
import numpy as np
import random
import torch.nn.functional as F


class DirectAUModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 learning_rate: float,
                 embed_k: int,
                 weight_decay: float,
                 gamma: float,
                 random_seed: int,
                 name="DirectAU",
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_normal_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_normal_(self.Gi.weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, inputs, **kwargs):
        users, items = inputs

        gamma_u = torch.squeeze(self.Gu.weight[users, :]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items, :]).to(self.device)

        return gamma_u, gamma_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu.weight[start:stop].to(self.device),
                            torch.transpose(self.Gi.weight.to(self.device), 0, 1))

    def train_step(self, batch):
        user, pos, _ = batch
        e_u, e_i = self.forward(inputs=(user, pos))  # user and positive item embeddings within the batch
        align = self.alignment(e_u, e_i)
        uniform = 0.5 * (self.uniformity(e_u) + self.uniformity(e_i))
        loss = align + self.gamma * uniform

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)