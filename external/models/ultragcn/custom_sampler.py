import numpy as np
import random
import torch
import torch.utils.data as data


class Sampler:
    def __init__(self, edge_index, num_items, interacted_items, negative_num, batch_size, sampling_sift_pos, seed=42):
        '''

        :param edge_index: list of tuples (user_id, item_id) representing the positive interactions (training data). Shape: (num_interactions, 2).
        :param num_items
        :param interacted_items: list of lists, where interacted_items[u] contains the item IDs that user u has interacted with. Used for filtering positive items during negative sampling if sampling_sift_pos is True. Outer list length: num_users. Inner lists have variable lengths.
        :param negative_num: number of negative items to sample for each positive interaction
        :param batch_size:  number of positive interactions to include in each training batch. Scalar integer.
        :param sampling_sift_pos: Boolean flag. If True, negative samples for a user u will explicitly exclude items already interacted with by u. If False, any item can be sampled as negative (including positives).
        :param seed:
        '''
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.num_items = num_items
        self.negative_num = negative_num
        self.sampling_sift_pos = sampling_sift_pos
        self.interacted_items = interacted_items ## list of list. interacted_items[user_id] = list of items interacted by user user_id
        # e.g. interacted_items[11] = [ 98, 47, 64] means that user 11 interacted with items 98, 47 and 64

        #  The DataLoader itself only yields batches of positive user-item pairs. The negative sampling happens after a batch is drawn from the loader, within the step method.
        self.train_loader = data.DataLoader(
            dataset=edge_index,
            batch_size=batch_size,
            shuffle=True
        )

    def step(self, pos_train_data):
        # Takes a batch of positive interactions from the DataLoader and performs negative sampling for them.
        # pos_tran_data = list of 2 elements: 2 torch.tensor of shape [batch_size,] - i.e. it comes from for x in train_loader.
        # pos_train_data[0] = Tensor of user IDs in the batch. Shape: (batch_size, )
        # pos_train_data[1] = Tensor of item Positive item IDs in the batch. Shape: (batch_size, )
        neg_candidates = np.arange(self.num_items)  # Creates a NumPy array containing all possible item IDs

        # If true: filter => Filter Positive, otherwhise random sampling
        if self.sampling_sift_pos:
            neg_items = []
            for u in pos_train_data[0]:
                # Create an initial probability array (all items equally likely). Shape: (num_items,).
                probs = np.ones(self.num_items)
                #  Set the probability of sampling items already interacted with by user u to zero.
                probs[self.interacted_items[u]] = 0
                # Normalize the probabilities so they sum to 1.
                probs /= np.sum(probs)
                # Sample self.negative_num items for user u using the calculated probs. replace=True allows
                # sampling the same negative item multiple times for a single user within a batch.
                # Reshape to (1, self.negative_num).
                u_neg_items = np.random.choice(neg_candidates, size=self.negative_num, p=probs, replace=True).reshape(1,-1)
                # Combine the lists of negative items for all users in the batch into a single NumPy array.
                # Shape: (batch_size, self.negative_num).
                neg_items.append(u_neg_items)

            neg_items = np.concatenate(neg_items, axis=0) # (batch_size, self.negative_num).
        else:
            neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), self.negative_num), replace=True)

        neg_items = torch.from_numpy(neg_items)  # convert it into torch.tensor

        # pos_train_data[0].long(): User IDs for the batch. Tensor, Shape: (batch_size,), Type: Long.
        # pos_train_data[1].long(): Positive Item IDs for the batch. Tensor, Shape: (batch_size,), Type: Long.
        # neg_items.long(): Sampled Negative Item IDs for the batch. Tensor, Shape: (batch_size, self.negative_num), Type: Long.
        return pos_train_data[0].long(), pos_train_data[1].long(), neg_items.long()  # users, pos_items, neg_items
