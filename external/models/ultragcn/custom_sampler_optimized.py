
import torch
import numpy as np
import torch.utils.data as data
class Sampler:
    def __init__(self, edge_index, num_items, interacted_items,
                 negative_num, batch_size, sampling_sift_pos, seed=42):
        """
        Initializes the Sampler for training with negative sampling.

        :param edge_index: Iterable of (user_id, item_id) pairs representing positive interactions.
                           This will be the Dataset for the DataLoader.
        :param num_items: Total number of unique items in the system (0..num_items-1).
        :param interacted_items: List of lists, where interacted_items[u] is a Python list of item IDs
                                 that user `u` has interacted with. Used to exclude positives during
                                 negative sampling if sampling_sift_pos=True.
        :param negative_num: Number of negative samples to draw per positive interaction.
        :param batch_size: Number of positive interactions per batch.
        :param sampling_sift_pos: If True, negative samples for each user will exclude items
                                  the user has already interacted with. If False, negatives
                                  may include previously seen items.
        :param seed: Random seed for reproducibility (affects both NumPy & PyTorch samplers).
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_items = num_items
        self.negative_num = negative_num
        self.sampling_sift_pos = sampling_sift_pos
        self.interacted_items = interacted_items
        # If we need to filter out positives from negatives, precompute for each user
        if self.sampling_sift_pos:
            # Precompute allowed negatives for each user
            # Create a 1D NumPy array of all item IDs [0, 1, ..., num_items-1]
            all_items = np.arange(num_items)
            # Create a 1D NumPy array of all item IDs [0, 1, ..., num_items-1]
            self.allowed_negatives = [
                np.setdiff1d(all_items, interacted_items[u], assume_unique=True)
                for u in range(len(interacted_items))
            ]

        self.train_loader = data.DataLoader(
            dataset=edge_index,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

    def step(self, pos_train_data):
        """
        Given one batch of positive (user, item) pairs, sample negatives.

        :param pos_train_data: Tuple (users, pos_items), each a LongTensor of shape (batch_size,).
        :returns: users, pos_items, neg_items where
                  - users:    LongTensor (batch_size,)
                  - pos_items:LongTensor (batch_size,)
                  - neg_items:LongTensor (batch_size, negative_num)
        """
        users, pos_items = pos_train_data
        batch_size = users.size(0)

        if self.sampling_sift_pos:
            # # Raster through each user in the batch and sample from their precomputed allowed list.
            # Complexity: O(batch_size × negative_num), no inner loop over num_items.
            # Only O(batch_size × negative_num) instead of O(batch_size * num_items)

            # Randomly choose `negative_num` items from allowed_negatives[u_idx]
            # with replacement (so duplicates within the same user are possible).
            neg_list = [
                np.random.choice(
                    self.allowed_negatives[u.item()],
                    self.negative_num,
                    replace=True
                )
                for u in users
            ]
            neg_items = torch.from_numpy(np.stack(neg_list, axis=0))
        else:
            # Entirely on GPU if desired
            neg_items = torch.randint(
                low=0,
                high=self.num_items,
                size=(batch_size, self.negative_num),
                device=users.device
            )
        # Return all tensors as Long (int64) for embedding lookups, etc.
        return users.long(), pos_items.long(), neg_items.long()