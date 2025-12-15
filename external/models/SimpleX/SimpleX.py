__version__ = '1.0.0'
__author__ = ''
__email__ = ''

import torch
import os
from tqdm import tqdm
import math

from elliot.dataset.samplers import custom_sampler as cs
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from .SimpleXModel import SimpleXModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger
from .multi_sampler import MultiSampler


class SimpleX(RecMixin, BaseRecommenderModel):
    r"""
    Implementation of the model SimpleX - paper: SimpleX: A Simple and Strong Baseline for Collaborative Filtering


    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        SimpleX: A Simple and Strong Baseline for Collaborative Filtering

        * This implements only the average pooling aggregation ( EQ 3 )*
         << average-pooling is a robust aggregation method that always demands a first attempt
        when applying SimpleX. The others  usually needs more efforts
        to tune and in some cases brings marginal improvements.>>

        (see <--paper link--> for details about the algorithm design choices).


        params:
        model parameters {factors: embedding size - d in the proposed paper
                        lr: learning rate
                        lw: embedding regularizer ( L2 loss  weight )
                         m: margin to filter negative samples in cosine contrastive loss (CCL) - Eq. (1)
                         nw: negative weight, control the relative weights of positive-sample loss and negative
                            one - Eq. (1)
                         g: fusion weight controls the balance between a user’s latent embedding and the
                            aggregated representation derived from their interaction history- Eq. (5)
                         dprob: dropout probability - not in the paper, but in the original released code.

                        }

        sampler parameters { s_s_p: If True, negative samples for a user u will explicitly exclude items
                                already interacted with by u. If False, any item can be sampled as negative
                            n_n: number of negative items
                            }

        """

        self._params_list = [
            ("_factors", "factors", "factors", 64, int, None),
            ("_lr", "lr", "lr", 0.001, float, None),
            ("_lw", "lw", "lw", 0.00001, float, None),
            ("_m", "m", "m", 0.9, float, None),
            ("_nw", "nw", "nw", 150, float, None),
            ("_g", "g", "g", 1.0, float, None),
            ("_dprob", "dprob", "dprob", 0.1, float, None),
            ("_s_s_p", "s_s_p", "s_s_p", False, bool, None),
            ("_n_n", "n_n", "n_n", 100, int, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        # self._sampler = cs.Sampler(self._data.i_train_dict, self._seed)

        # Interaction matrix - it includes the mean operator
        interaction_matrix = self.get_normalized_matrix(self._data.i_train_dict, self._num_users, self._num_items)

        row, col = data.sp_i_train.nonzero()
        edge_index = list(zip(row, col))

        interacted_items = [[] for _ in range(self._num_users)]
        for (u, i) in edge_index:
            interacted_items[u].append(i)

        self._sampler = MultiSampler(edge_index=edge_index,
                                     num_items=self._num_items,
                                     batch_size=self._batch_size,
                                     negative_num=self._n_n,
                                     sampling_sift_pos=self._s_s_p,
                                     interacted_items=interacted_items,
                                     seed=self._seed)

        self._model = SimpleXModel(self._num_users,
                                   self._num_items,
                                   interaction_matrix,
                                   self._factors,
                                   self._lr,
                                   self._lw,
                                   self._m,
                                   self._nw,
                                   self._g,
                                   self._dprob,
                                   self._seed)

    @property
    def name(self):
        return "SimpleX" \
            + f"_{self.get_base_params_shortcut()}" \
            + f"_{self.get_params_shortcut()}"

    @staticmethod
    def get_normalized_matrix(interactions: dict, num_users: int, num_items: int) -> torch.Tensor:

        # Initialize the interaction matrix
        interaction_matrix = torch.zeros((num_users, num_items))

        # Compute user_degree N(u)  and item_degree N(i)
        user_degrees = torch.zeros(num_users)
        item_degrees = torch.zeros(num_items)
        for user_id, items_dict in interactions.items():
            user_degrees[user_id] = len(items_dict)
            for item_id in items_dict.keys():
                item_degrees[item_id] += 1

        # Prevent division by zero
        user_degrees[user_degrees == 0] = 1.0
        item_degrees[item_degrees == 0] = 1.0

        # Populate the interaction matrix according to the specific scaling
        for user_id, items_dict in interactions.items():
            for item_id in items_dict.keys():
                # mean over user
                interaction_matrix[user_id, item_id] = 1.0 / (user_degrees[user_id])
        return interaction_matrix

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            n_batch = int(
                self._data.transactions / self._batch_size) if self._data.transactions % self._batch_size == 0 else int(
                self._data.transactions / self._batch_size) + 1
            with tqdm(total=n_batch, disable=not self._verbose) as t:
                for x in self._sampler.train_loader:
                    steps += 1
                    users, pos_items, neg_items = self._sampler.step(x)
                    batch_loss = self._model.train_step((users,
                                                         pos_items,
                                                         neg_items))
                    loss += batch_loss.item() if isinstance(batch_loss, torch.Tensor) else float(batch_loss)

                    if math.isnan(loss) or math.isinf(loss) or (not loss):
                        break

                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
