import os
import math
import pandas as pd
import numpy as np
from collections import Counter

# graph libraries
import networkx
from networkx.algorithms import bipartite


#### This class is used to compute dataset characteristics
class RecommendationDataset:

    def __init__(self, data: pd.DataFrame, user_col=None, item_col=None, rating_col=None, timestamp_col=None):

        assert isinstance(data, pd.DataFrame)
        self._dataset = data
        self._is_private = False

        n_columns = len(data.columns)
        assert n_columns >= 2, f'{self.__class__.__name__}: the columns must be at least two.'

        # user and item column
        if user_col is None:
            user_col = 0
        if item_col is None:
            item_col = 1

        # if ratings column is missing, it is assumed that ratings are implicit
        self._is_implicit = False
        if (len(data.columns)) < 3 and (rating_col is None):
            rating_col = 'ratings'
            data[rating_col] = [1] * len(data)
            self._is_implicit = True

        self._names = data.columns.to_list()
        self.user_col = None
        self.item_col = None
        self.rating_col = None
        self.timestamp_col = None
        self.set_columns(user_col, item_col, rating_col, timestamp_col)

        # -- users and items setting --
        self._sorted_users = None
        self._sorted_items = None

        # map users and items with a 0-indexed mapping
        self._public_to_private_users = self.public_to_private(self._dataset[self.user_col].unique().tolist())
        self._n_users = len(self._public_to_private_users)

        self._public_to_private_items = self.public_to_private(self._dataset[self.item_col].unique().tolist(),
                                                               self._n_users)
        self._n_items = len(self._public_to_private_items)

        self._private_to_public_users = self.private_to_public(self._public_to_private_users)
        self._private_to_public_items = self.private_to_public(self._public_to_private_items)

        self.to_private()

        self._users = list(range(self._n_users))
        # self._items = list(range(self._n_items))
        self._items = list(range(self._n_users, self._n_users + self._n_items))

        # -- ratings setting --

        # metrics
        self._transactions = None
        self._space_size = None
        self._space_size_log = None
        self._shape = None
        self._shape_log = None
        self._density = None
        self._density_log = None
        self._gini_item = None
        self._gini_user = None
        self.metrics = ['transactions', 'space_size', 'space_size_log', 'shape', 'shape_log', 'density', 'density_log',
                        'gini_item', 'gini_user',
                        'average_degree', 'average_degree_users', 'average_degree_items',
                        'average_degree_log', 'average_degree_users_log', 'average_degree_items_log',
                        'average_clustering_coefficient_dot',
                        'average_clustering_coefficient_min',
                        'average_clustering_coefficient_max',
                        'average_clustering_coefficient_dot_log',
                        'average_clustering_coefficient_min_log',
                        'average_clustering_coefficient_max_log',
                        'average_clustering_coefficient_dot_users', 'average_clustering_coefficient_dot_items',
                        'average_clustering_coefficient_min_users', 'average_clustering_coefficient_min_items',
                        'average_clustering_coefficient_max_users', 'average_clustering_coefficient_max_items',
                        'average_clustering_coefficient_dot_users_log', 'average_clustering_coefficient_dot_items_log',
                        'average_clustering_coefficient_min_users_log', 'average_clustering_coefficient_min_items_log',
                        'average_clustering_coefficient_max_users_log', 'average_clustering_coefficient_max_items_log',
                        'degree_assortativity_users', 'degree_assortativity_items', 'most_least_favorite', 'most_rated',
                        'highest_variance',
                        'most_recent_1', 'most_recent_2', 'most_recent_3',
                        'most_characteristic']
        self._ratings_per_user = None
        self._ratings_per_item = None

    @property
    def n_users(self):
        self._n_users = int(self._dataset[self.user_col].nunique())
        return self._n_users

    @property
    def n_items(self):
        self._n_items = int(self._dataset[self.item_col].nunique())
        return self._n_items

    def set_columns(self, user_col, item_col, rating_col, timestamp_col):

        new_col_names = {col: name for col, name in zip([user_col, item_col, rating_col, timestamp_col],
                                                        ['user', 'item', 'rating', 'timestamp'])}
        self._dataset.rename(columns=new_col_names, inplace=True)

        for var, col in zip(['user_col', 'item_col', 'rating_col', 'timestamp_col'], self._dataset.columns):
            self.__setattr__(var, col)

        # if self.timestamp_col:
        #     if not np.issubdtype(self._dataset[self.timestamp_col].dtype, np.number):
        #         # Se è datetime, converte in timestamp Unix in secondi
        #         self._dataset[self.timestamp_col] = pd.to_datetime(self._dataset[self.timestamp_col]).astype(
        #             np.int64) // 10 ** 9
        #     elif self._dataset[self.timestamp_col].max() > 1e12:
        #         # Se già numerico, ma in millisecondi (valori molto grandi), converte in secondi
        #         self._dataset[self.timestamp_col] = (self._dataset[self.timestamp_col] // 1000).astype(int)

    @property
    def sorted_items(self):
        if self._sorted_items is None:
            count_items = self._dataset.groupby(self.item_col).count().sort_values(by=[self.user_col])
            self._sorted_items = dict(zip(count_items.index, count_items[self.user_col]))
        return self._sorted_items

    @property
    def sorted_users(self):
        if self._sorted_users is None:
            count_users = self._dataset.groupby(self.user_col).count().sort_values(by=[self.item_col])
            self._sorted_users = dict(zip(count_users.index, count_users[self.item_col]))
        return self._sorted_users

    # --- MAPPING FUNCTIONS ---
    @property
    def transactions(self):
        if self._transactions is None:
            self._transactions = len(self._dataset)
        return self._transactions

    @staticmethod
    def public_to_private(lst, offset=0):
        return {el: idx + offset for idx, el in enumerate(lst)}

    @staticmethod
    def private_to_public(pub_to_prvt: dict):
        mapping = {el: idx for idx, el in pub_to_prvt.items()}
        if len(pub_to_prvt) != len(mapping):
            print('WARNING: private to public mapping could be incorrect. Please, check your code.')
        return mapping

    def map_dataset(self, user_mapping, item_mapping):
        self._dataset[self.user_col] = self._dataset[self.user_col].map(user_mapping)
        self._dataset[self.item_col] = self._dataset[self.item_col].map(item_mapping)

    def to_public(self):
        if self._is_private:
            self.map_dataset(self._private_to_public_users, self._private_to_public_items)
        self._is_private = False

    def to_private(self):
        if not self._is_private:
            self.map_dataset(self._public_to_private_users, self._public_to_private_items)
        self._is_private = True

    def to_csv(self, path, **kwargs):
        self._dataset.to_csv(path, **kwargs)

    # -- METRICS --

    def get_metric(self, metric):
        assert metric in self.metrics, f'{self.__class__.__name__}: metric \'{metric}\' not found.'
        func = getattr(self, metric)
        return func()

    def space_size(self):
        if self._space_size is None:
            scale_factor = 1000
            self._space_size = math.sqrt(self._n_users * self._n_items) / scale_factor
        return self._space_size

    def space_size_log(self):
        if self._space_size_log is None:
            self._space_size_log = math.log10(self.space_size())
        return self._space_size_log

    def shape(self):
        if self._shape is None:
            self._shape = self._n_users / self._n_items
        return self._shape

    def shape_log(self):
        if self._shape_log is None:
            self._shape_log = math.log10(self.shape())
        return self._shape_log

    def density(self):
        if self._density is None:
            self._density = self.transactions / (self._n_users * self._n_items)
        return self._density

    def density_log(self):
        if self._density_log is None:
            self._density_log = math.log10(self.density())
        return self._density_log

    @staticmethod
    def gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x) ** 2 * np.mean(x))

    def gini_item(self):
        if self._gini_item is None:
            self._gini_item = self.gini(np.array(list(self.sorted_items.values())))
        return self._gini_item

    def gini_user(self):
        if self._gini_user is None:
            self._gini_user = self.gini(np.array(list(self.sorted_users.values())))
        return self._gini_user

    def ratings_per_user(self):

        if self._ratings_per_user is None:
            self._ratings_per_user = self.transactions / self.n_users
        return self._ratings_per_user

    def ratings_per_item(self):

        if self._ratings_per_item is None:
            self._ratings_per_item = self.transactions / self.n_items
        return self._ratings_per_item


class GraphDataset(RecommendationDataset):

    def __init__(self, data: pd.DataFrame):
        assert isinstance(data, pd.DataFrame)

        super(GraphDataset, self).__init__(data)

        self.bipartite_graph = self.networkx_bipartite_graph()
        self.num_edges = len(self.bipartite_graph.edges)

        all_users_in_dataset = set(self._dataset[self.user_col].unique())
        all_items_in_dataset = set(self._dataset[self.item_col].unique())

        if not all_users_in_dataset.issubset(self._users):
            missing_users = all_users_in_dataset - set(self._users)
            raise ValueError(f"Utenti nel dataset non presenti in self._users: {missing_users}")

        if not all_items_in_dataset.issubset(self._items):
            missing_items = all_items_in_dataset - set(self._items)
            raise ValueError(f"Items nel dataset non presenti in self._items: {missing_items}")


        self.user_nodes = {n for n in self.bipartite_graph.nodes if
                           self.bipartite_graph.nodes[n]['bipartite'] == 'users'}

        self.item_nodes = {n for n in self.bipartite_graph.nodes if
                           self.bipartite_graph.nodes[n]['bipartite'] == 'items'}

        for n in self.bipartite_graph.nodes:

            if 'bipartite' not in self.bipartite_graph.nodes[n]:
                raise ValueError(f"Nodo {n} senza attributo 'bipartite'.")



        self.num_user_nodes, self.num_item_nodes = len(self.user_nodes), len(self.item_nodes)
        self.user_graph, self.item_graph = None, None

        # metrics

        # degree
        self._average_degree = None
        self._average_degree_users = None
        self._average_degree_items = None
        self._average_degree_log = None
        self._average_degree_users_log = None
        self._average_degree_items_log = None

        # clustering
        self._average_clustering_coefficient_dot = None
        self._average_clustering_coefficient_min = None
        self._average_clustering_coefficient_max = None
        self._average_clustering_coefficient_dot_users = None
        self._average_clustering_coefficient_dot_items = None
        self._average_clustering_coefficient_min_users = None
        self._average_clustering_coefficient_min_items = None
        self._average_clustering_coefficient_max_users = None
        self._average_clustering_coefficient_max_items = None
        self._average_clustering_coefficient_dot_log = None
        self._average_clustering_coefficient_min_log = None
        self._average_clustering_coefficient_max_log = None
        self._average_clustering_coefficient_dot_users_log = None
        self._average_clustering_coefficient_dot_items_log = None
        self._average_clustering_coefficient_min_users_log = None
        self._average_clustering_coefficient_min_items_log = None
        self._average_clustering_coefficient_max_users_log = None
        self._average_clustering_coefficient_max_items_log = None

        # assortativity
        self._degree_assortativity = None
        self._degree_assortativity_users = None
        self._degree_assortativity_items = None

        # data minimization
        self._most_least_favorite = None
        self._most_rated = None
        self._highest_variance = None
        self._most_recent_1 = None
        self._most_recent_2 = None
        self._most_recent_3 = None
        self._most_characteristic = None

    def average_degree(self):

        if self._average_degree is None:
            self._average_degree = (2 * self.num_edges) / (self.num_user_nodes + self.num_item_nodes)

        return self._average_degree

    def average_degree_log(self):

        if self._average_degree_log is None:
            self._average_degree_log = math.log10(self.average_degree())

        return self._average_degree_log

    def average_degree_users(self):

        if self._average_degree_users is None:
            self._average_degree_users = self.num_edges / self.num_user_nodes

        return self._average_degree_users

    def average_degree_users_log(self):

        if self._average_degree_users_log is None:
            self._average_degree_users_log = math.log10(self.average_degree_users())

        return self._average_degree_users_log

    def average_degree_items(self):

        if self._average_degree_items is None:
            self._average_degree_items = self.num_edges / self.num_item_nodes

        return self._average_degree_items

    def average_degree_items_log(self):

        if self._average_degree_items_log is None:
            self._average_degree_items_log = math.log10(self.average_degree_items())

        return self._average_degree_items_log

    def average_clustering_coefficient_dot(self):

        if self._average_clustering_coefficient_dot is None:
            self._average_clustering_coefficient_dot = bipartite.average_clustering(self.bipartite_graph,
                                                                                    mode='dot')
        return self._average_clustering_coefficient_dot

    def average_clustering_coefficient_dot_log(self):

        if self._average_clustering_coefficient_dot_log is None:
            self._average_clustering_coefficient_dot_log = math.log10(self.average_clustering_coefficient_dot())
        return self._average_clustering_coefficient_dot_log

    def average_clustering_coefficient_min(self):

        if self._average_clustering_coefficient_min is None:
            self._average_clustering_coefficient_min = bipartite.average_clustering(self.bipartite_graph,
                                                                                    mode='min')
        return self._average_clustering_coefficient_min

    def average_clustering_coefficient_min_log(self):

        if self._average_clustering_coefficient_min_log is None:
            self._average_clustering_coefficient_min_log = math.log10(self.average_clustering_coefficient_min())
        return self._average_clustering_coefficient_min_log

    def average_clustering_coefficient_max(self):

        if self._average_clustering_coefficient_max is None:
            self._average_clustering_coefficient_max = bipartite.average_clustering(self.bipartite_graph,
                                                                                    mode='max')
        return self._average_clustering_coefficient_max

    def average_clustering_coefficient_max_log(self):

        if self._average_clustering_coefficient_max_log is None:
            self._average_clustering_coefficient_max_log = math.log10(self.average_clustering_coefficient_max())
        return self._average_clustering_coefficient_max_log

    def average_clustering_coefficient_dot_users(self):

        if self._average_clustering_coefficient_dot_users is None:
            self._average_clustering_coefficient_dot_users = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='dot',
                                                                                          nodes=self.user_nodes)
        return self._average_clustering_coefficient_dot_users

    def average_clustering_coefficient_dot_users_log(self):

        if self._average_clustering_coefficient_dot_users_log is None:
            self._average_clustering_coefficient_dot_users_log = math.log10(
                self.average_clustering_coefficient_dot_users()
            )
        return self._average_clustering_coefficient_dot_users_log

    def average_clustering_coefficient_dot_items(self):

        if self._average_clustering_coefficient_dot_items is None:
            self._average_clustering_coefficient_dot_items = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='dot',
                                                                                          nodes=self.item_nodes)
        return self._average_clustering_coefficient_dot_items

    def average_clustering_coefficient_dot_items_log(self):

        if self._average_clustering_coefficient_dot_items_log is None:
            self._average_clustering_coefficient_dot_items_log = math.log10(
                self.average_clustering_coefficient_dot_items()
            )
        return self._average_clustering_coefficient_dot_items_log

    def average_clustering_coefficient_min_users(self):

        if self._average_clustering_coefficient_min_users is None:
            self._average_clustering_coefficient_min_users = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='min',
                                                                                          nodes=self.user_nodes)
        return self._average_clustering_coefficient_min_users

    def average_clustering_coefficient_min_users_log(self):

        if self._average_clustering_coefficient_min_users_log is None:
            self._average_clustering_coefficient_min_users_log = math.log10(
                self.average_clustering_coefficient_min_users()
            )
        return self._average_clustering_coefficient_min_users_log

    def average_clustering_coefficient_min_items(self):

        if self._average_clustering_coefficient_min_items is None:
            self._average_clustering_coefficient_min_items = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='min',
                                                                                          nodes=self.item_nodes)
        return self._average_clustering_coefficient_min_items

    def average_clustering_coefficient_min_items_log(self):

        if self._average_clustering_coefficient_min_items_log is None:
            self._average_clustering_coefficient_min_items_log = math.log10(
                self.average_clustering_coefficient_min_items()
            )
        return self._average_clustering_coefficient_min_items_log

    def average_clustering_coefficient_max_users(self):

        if self._average_clustering_coefficient_max_users is None:
            self._average_clustering_coefficient_max_users = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='max',
                                                                                          nodes=self.user_nodes)
        return self._average_clustering_coefficient_max_users

    def average_clustering_coefficient_max_users_log(self):

        if self._average_clustering_coefficient_max_users_log is None:
            self._average_clustering_coefficient_max_users_log = math.log10(
                self.average_clustering_coefficient_max_users()
            )
        return self._average_clustering_coefficient_max_users_log

    def average_clustering_coefficient_max_items(self):

        if self._average_clustering_coefficient_max_items is None:
            self._average_clustering_coefficient_max_items = bipartite.average_clustering(self.bipartite_graph,
                                                                                          mode='max',
                                                                                          nodes=self.item_nodes)
        return self._average_clustering_coefficient_max_items

    def average_clustering_coefficient_max_items_log(self):

        if self._average_clustering_coefficient_max_items_log is None:
            self._average_clustering_coefficient_max_items_log = math.log10(
                self.average_clustering_coefficient_max_items()
            )
        return self._average_clustering_coefficient_max_items_log

    def degree_assortativity_users(self):

        if self._degree_assortativity_users is None and self.user_graph is None:
            self.user_graph = bipartite.projected_graph(self.bipartite_graph, self.user_nodes)
            self._degree_assortativity_users = networkx.degree_pearson_correlation_coefficient(self.user_graph,
                                                                                               nodes=self.user_nodes)

        return self._degree_assortativity_users

    def degree_assortativity_items(self):

        if self._degree_assortativity_items is None and self.item_graph is None:
            self.item_graph = bipartite.projected_graph(self.bipartite_graph, self.item_nodes)
            self._degree_assortativity_items = networkx.degree_pearson_correlation_coefficient(self.item_graph,
                                                                                               nodes=self.item_nodes)

        return self._degree_assortativity_items

    ############ METODI AGGIUNTI

    # Most/Least Favorite Metric (normalized)
    def most_least_favorite(self):
        if self._most_least_favorite is None:
            user_avg_rating = self._dataset.groupby(self.user_col)[self.rating_col].mean()
            min_rating = self._dataset[self.rating_col].min()
            max_rating = self._dataset[self.rating_col].max()
            normalized_ratings = (user_avg_rating - min_rating) / (max_rating - min_rating)  # min-max normalization
            self._most_least_favorite = normalized_ratings.mean()
        return self._most_least_favorite

    # Most Rated Metric (normalized)
    def most_rated(self):
        if self._most_rated is None:
            item_degree = self._dataset[self.item_col].value_counts()
            max_degree = item_degree.max()
            normalized_degree = item_degree / max_degree
            user_avg_degree = self._dataset[self.item_col].map(normalized_degree).groupby(
                self._dataset[self.user_col]).mean()
            self._most_rated = user_avg_degree.mean()
        return self._most_rated

    # Highest Variance Metric (normalized)
    def highest_variance(self):
        if self._highest_variance is None:
            rating_range = self._dataset[self.rating_col].max() - self._dataset[self.rating_col].min()
            item_std = self._dataset.groupby(self.item_col)[self.rating_col].std().fillna(0)
            normalized_std = item_std / rating_range
            user_avg_std = self._dataset[self.item_col].map(normalized_std).groupby(self._dataset[self.user_col]).mean()
            self._highest_variance = user_avg_std.mean()
        return self._highest_variance

    # Most Recent Metric 1 (mean timestamp normalized)
    def most_recent_1(self):
        if self._most_recent_1 is None:
            max_timestamp = self._dataset[self.timestamp_col].max()
            user_mean_ts = self._dataset.groupby(self.user_col)[self.timestamp_col].mean() / max_timestamp
            self._most_recent_1 = user_mean_ts.mean()
        return self._most_recent_1

    # Most Recent Metric 2 (max timestamp normalized)
    def most_recent_2(self):
        if self._most_recent_2 is None:
            max_timestamp = self._dataset[self.timestamp_col].max()
            user_max_ts = self._dataset.groupby(self.user_col)[self.timestamp_col].max() / max_timestamp
            self._most_recent_2 = user_max_ts.mean()
        return self._most_recent_2

    # Most Recent Metric 3 (normalized delta per user)
    def most_recent_3(self):
        if self._most_recent_3 is None:
            def user_normalized_delta(x):
                max_ts = x.max()
                min_ts = x.min()
                if max_ts == min_ts:
                    return 0
                return ((max_ts - x).mean()) / (max_ts - min_ts)

            user_delta = self._dataset.groupby(self.user_col)[self.timestamp_col].apply(user_normalized_delta)
            self._most_recent_3 = user_delta.mean()
        return self._most_recent_3

    # Most Characteristic Metric (normalized)
    def most_characteristic(self):
        if self._most_characteristic is None:
            user_item_matrix = pd.crosstab(self._dataset[self.item_col], self._dataset[self.user_col])
            avg_vector = user_item_matrix.mean(axis=0).to_numpy()
            distances = np.linalg.norm(user_item_matrix.values - avg_vector, axis=1)
            max_distance = distances.max()
            normalized_distances = distances / max_distance

            item_distance_map = dict(zip(user_item_matrix.index, normalized_distances))
            user_avg_dist = self._dataset[self.item_col].map(item_distance_map).groupby(
                self._dataset[self.user_col]).mean()
            self._most_characteristic = user_avg_dist.mean()
        return self._most_characteristic

    def networkx_bipartite_graph(self):
        # build undirected and bipartite graph with networkx
        print(f'{self.__class__.__name__}: building a bipartite graph with networkx')
        graph = networkx.Graph()
        graph.add_nodes_from(self._users, bipartite='users')
        graph.add_nodes_from(self._items, bipartite='items')
        graph.add_edges_from(list(zip(self._dataset[self.user_col], self._dataset[self.item_col])))
        return graph

    def connected_graph(self):
        graph = self.networkx_bipartite_graph()
        # if graph is not connected, retain only the biggest connected portion
        if not networkx.is_connected(graph):
            print(f'{self.__class__.__name__}: the graph is not connected. Building the connected subgraph.')
            graph = graph.subgraph(max(networkx.connected_components(graph), key=len))
        print(f'{self.__class__.__name__}: graph is connected.')
        user_nodes, item_nodes = bipartite.sets(graph)
        print(f'{self.__class__.__name__}: graph characteristics'
              f'\n nodes: {len(graph.nodes)}'
              f'\n edges: {len(graph.edges)}'
              f'\n user nodes: {len(user_nodes)}'
              f'\n item nodes: {len(item_nodes)}')
        return graph

    def make_connected(self):
        self.bipartite_graph = self.connected_graph()

        user_nodes, item_nodes = bipartite.sets(self.bipartite_graph)

        n_old_users = self._n_users
        n_old_items = self._n_items

        # filter out users and items
        self._dataset = self._dataset[
            self._dataset[self.user_col].isin(user_nodes) & self._dataset[self.item_col].isin(item_nodes)]
        assert user_nodes == set(self._dataset[self.user_col].unique()), f'{self.__class__.__name__}:' \
                                                                         f' a problem occurred during dataset filtering'
        assert item_nodes == set(self._dataset[self.item_col].unique()), f'{self.__class__.__name__}:' \
                                                                         f' a problem occurred during dataset filtering'

        self._n_users = len(user_nodes)
        self._n_items = len(item_nodes)

        print(f'{self.__class__.__name__}: {n_old_users - self._n_users} users removed')
        print(f'{self.__class__.__name__}: {n_old_items - self._n_items} items removed')

    def compute_all_characteristics(self, dataset_name=None, cutoff: int = 0, strategy: str = 'None'):
        data = {}
        if dataset_name is not None:
            data['dataset_name'] = dataset_name
            data['strategy'] = strategy
            data['cutoff'] = cutoff
        for metric in self.metrics:
            print(f"\t\t .. computing {metric}")
            try:
                metric_attr = getattr(self.__class__, metric, None)
                if isinstance(metric_attr, property):
                    data[metric] = getattr(self, metric)
                else:
                    metric_func = getattr(self, metric)
                    data[metric] = metric_func()
            except Exception as e:
                data[metric] = float('nan')
        return pd.DataFrame([data])

    def compute_timestamp_characteristic(self, dataset_name=None, cutoff: int = 0, strategy: str = 'None'):
        data = {}
        if dataset_name is not None:
            data['dataset_name'] = dataset_name
            data['strategy'] = strategy
            data['cutoff'] = cutoff
        metrics = ['most_recent_1', 'most_recent_2', 'most_recent_3']
        for metric in metrics:
            print(f"\t\t .. computing {metric}")
            try:
                metric_attr = getattr(self.__class__, metric, None)
                if isinstance(metric_attr, property):
                    data[metric] = getattr(self, metric)
                else:
                    metric_func = getattr(self, metric)
                    data[metric] = metric_func()
            except Exception as e:
                data[metric] = float('nan')
        return pd.DataFrame([data])