# custom_sampler.py

import torch
import torch.utils.data as data


class MultiSampler:
    def __init__(self,
                 edge_index,  # list of (user, item) tuples
                 num_items: int,
                 interacted_items,  # list of lists: interacted_items[u] = [i1, i2, …]
                 negative_num: int,
                 batch_size: int,
                 sampling_sift_pos: bool,
                 seed: int = 42):
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_items = num_items
        self.negative_num = negative_num
        self.sampling_sift_pos = sampling_sift_pos

        # If we need to exclude all past positives, build a padded "allowed negatives" matrix
        if sampling_sift_pos:
            num_users = len(interacted_items)
            # first compute lengths
            lengths = []
            rows = []  # will accumulate rows of allowed ids
            for u in range(num_users):
                mask = torch.ones(num_items,
                                  dtype=torch.bool)  # Creates a boolean mask of size num_items, initially all True.
                # Accesses the list of items user u has interacted with (interacted_items[u]) and sets the corresponding
                # positions in the mask to False. Now, mask[i] is True only if user u has not interacted with item i.
                mask[interacted_items[u]] = False
                # Finds the indices where the mask is True (i.e., the item IDs that are allowed negatives for user u)
                # and converts them into a 1D tensor negs.
                negs = mask.nonzero(as_tuple=False).view(-1)
                #  Stores the number of allowed negative items for user u.
                lengths.append(negs.size(0))
                rows.append(negs)  # Stores the tensor containing the actual allowed negative item IDs for user u.
            max_len = max(lengths)
            # build matrix and length tensor
            # Creates a tensor (num_users rows, max_len columns) initialized with zeros.
            # This will hold the allowed negative item IDs for all users, padded with zeros where necessary.
            allowed_matrix = torch.zeros((num_users, max_len), dtype=torch.long)
            #  Creates a tensor storing the actual length of the allowed negatives list for each user.
            #  This is crucial later to know how many valid entries exist in each row of allowed_matrix.
            length_tensor = torch.tensor(lengths, dtype=torch.long)
            # fill allowed madrix
            for u, negs in enumerate(rows):
                '''Iterates through the collected allowed negative tensors (rows).
                L = negs.size(0): Gets the actual number of allowed negatives for the current user u.
                allowed_matrix[u, :L] = negs: Copies the allowed negative item IDs (negs) into the first L columns of the corresponding row (u) in allowed_matrix. The remaining columns (L to max_len-1) stay zero.
                '''
                L = negs.size(0)
                allowed_matrix[u, :L] = negs
            self.allowed_matrix = allowed_matrix.to(self.device)
            self.allowed_lengths = length_tensor.to(self.device)
        else:
            self.allowed_matrix = None
            self.allowed_lengths = None

        # pack positive edges into a tensor
        edges = torch.tensor(edge_index, dtype=torch.long)
        self.train_loader = data.DataLoader(
            dataset=edges,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

    def step(self, batch_edges):
        """
        batch_edges: LongTensor[(B,2)] of (user, pos_item).
        returns: users (B,), pos_items (B,), neg_items (B, negative_num)
        """
        users = batch_edges[:, 0].to(self.device)
        pos_items = batch_edges[:, 1].to(self.device)
        B = users.size(0)
        N = self.negative_num

        if self.sampling_sift_pos:
            # for each user, sample uniformly from allowed_matrix[u, :lengths[u]]
            lengths = self.allowed_lengths[users]  # (B,)
            # sample floats in [0,1) then scale to [0, lengths)
            randf = torch.rand((B, N), device=self.device)
            idx = (randf * lengths.unsqueeze(1).float()).long()  # (B,N)
            cand = self.allowed_matrix[users]  # (B, max_len)
            neg_items = cand.gather(1, idx)  # (B,N)
        else:
            neg_items = torch.randint(0, self.num_items, (B, N), device=self.device)

        return users, pos_items, neg_items
