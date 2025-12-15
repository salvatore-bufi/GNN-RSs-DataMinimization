import os
import pandas as pd
import numpy as np
import random
import networkx
import torch
from networkx.algorithms import bipartite
import csv
from torch_geometric.utils.dropout import dropout_node, dropout_edge

# ==========================
# GLOBAL DEFAULT PARAMETERS
# ==========================
DATASETS = ['amazon-book', 'yelp']          # datasets to iterate over
DATASET_FILE_NAME = 'dm_candidate.tsv'      # input filename (same for all datasets)
SAMPLING_STRATEGIES = ['ND', 'ED']          # graph sampling strategies
NUM_SAMPLINGS = 900                         # number of samplings
START_IDX = 0                               # starting index
RANDOM_SEED = 42                            # base seed

DATASET_SCHEMAS = {
    'amazon-book': {
        'user_col': 'user_id',
        'item_col': 'parent_asin',
        'rating_col': 'rating',
        'timestamp_col': 'timestamp',
    },
    'yelp': {
        'user_col': 'user_id',
        'item_col': 'business_id',
        'rating_col': 'stars',
        'timestamp_col': 'date',
    },
}


def set_all_seeds(current_seed):
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    torch.cuda.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    torch.backends.cudnn.deterministic = True


def calculate_statistics_private(edge_index_private, num_users, num_items):
    """
    Compute statistics on private indices and, if the graph is disconnected,
    restrict to the largest connected component in a deterministic way.
    Returns: stats_dict, filtered_edge_index (or None if already connected).
    """
    g = networkx.Graph()
    g.add_nodes_from(range(num_users), bipartite='users')
    g.add_nodes_from(range(num_users, num_users + num_items), bipartite='items')
    edges_list = list(zip(edge_index_private[0].tolist(), edge_index_private[1].tolist()))
    g.add_edges_from(edges_list)

    if networkx.is_connected(g):
        user_nodes, item_nodes = bipartite.sets(g)
        m = g.number_of_edges()
        delta_g = m / (len(user_nodes) * len(item_nodes)) if len(user_nodes) and len(item_nodes) else 0.0
        stats_dict = {
            'users': len(user_nodes),
            'items': len(item_nodes),
            'interactions': m,
            'delta_g': delta_g
        }
        return stats_dict, None
    else:
        comps = list(networkx.connected_components(g))
        # deterministic tie-break: first by size, then by minimum node id
        comps.sort(key=lambda c: (len(c), min(c)), reverse=True)
        g_sub = g.subgraph(comps[0]).copy()

        user_nodes, item_nodes = bipartite.sets(g_sub)
        m = g_sub.number_of_edges()
        delta_g = m / (len(user_nodes) * len(item_nodes)) if len(user_nodes) and len(item_nodes) else 0.0
        stats_dict = {
            'users': len(user_nodes),
            'items': len(item_nodes),
            'interactions': m,
            'delta_g': delta_g
        }

        sub_edges = list(g_sub.edges())
        edge_index_sub = torch.tensor([[i for (i, j) in sub_edges],
                                       [j for (i, j) in sub_edges]], dtype=torch.int64)
        return stats_dict, edge_index_sub


def graph_sampling(dataset_name):
    schema = DATASET_SCHEMAS[dataset_name]
    user_col = schema['user_col']
    item_col = schema['item_col']
    rating_col = schema['rating_col']
    ts_col = schema['timestamp_col']

    # === 1) Load dataset with header ===
    path = f'./data/{dataset_name}/{DATASET_FILE_NAME}'
    dataset = pd.read_csv(path, sep='\t', header=0)

    # Minimal validation
    for c in [user_col, item_col, rating_col, ts_col]:
        if c not in dataset.columns:
            raise ValueError(f"Manca la colonna '{c}' nel file {path}")

    # === 2) Reindex public -> private while keeping all columns ===
    initial_users = dataset[user_col].unique().tolist()
    initial_items = dataset[item_col].unique().tolist()
    initial_num_users = len(initial_users)
    initial_num_items = len(initial_items)

    public_to_private_users = {u: idx for idx, u in enumerate(initial_users)}
    public_to_private_items = {i: idx + initial_num_users for idx, i in enumerate(initial_items)}

    dataset['user_idx'] = dataset[user_col].map(public_to_private_users).astype('int64')
    dataset['item_idx'] = dataset[item_col].map(public_to_private_items).astype('int64')

    # Build initial graph (in private indices)
    g = networkx.Graph()
    g.add_nodes_from(range(initial_num_users), bipartite='users')
    g.add_nodes_from(range(initial_num_users, initial_num_users + initial_num_items), bipartite='items')
    g.add_edges_from(list(zip(dataset['user_idx'].tolist(), dataset['item_idx'].tolist())))

    if not networkx.is_connected(g):
        comps = list(networkx.connected_components(g))
        comps.sort(key=lambda c: (len(c), min(c)), reverse=True)
        biggest = comps[0]
        g_sub = g.subgraph(biggest).copy()
        sub_edges = set(g_sub.edges())

        # Keep only rows whose (user_idx, item_idx) pair is in the largest component
        dataset = dataset[dataset.apply(lambda r: (r['user_idx'], r['item_idx']) in sub_edges, axis=1)].copy()

        # Recompute mappings on the sub-dataset
        connected_users = dataset[user_col].unique().tolist()
        connected_items = dataset[item_col].unique().tolist()
        num_users = len(connected_users)
        num_items = len(connected_items)

        public_to_private_users = {u: idx for idx, u in enumerate(connected_users)}
        public_to_private_items = {i: idx + num_users for idx, i in enumerate(connected_items)}

        dataset['user_idx'] = dataset[user_col].map(public_to_private_users).astype('int64')
        dataset['item_idx'] = dataset[item_col].map(public_to_private_items).astype('int64')

        edge_index = torch.tensor([dataset['user_idx'].tolist(), dataset['item_idx'].tolist()], dtype=torch.int64)
    else:
        num_users = initial_num_users
        num_items = initial_num_items
        edge_index = torch.tensor([dataset['user_idx'].tolist(), dataset['item_idx'].tolist()], dtype=torch.int64)

    # Initial statistics
    m0 = edge_index.shape[1]
    delta_g0 = m0 / (num_users * num_items) if (num_users and num_items) else 0.0
    print(f'\n==============================')
    print(f'DATASET: {dataset_name}')
    print(f'Number of users: {num_users}')
    print(f'Number of items: {num_items}')
    print(f'Number of interactions: {m0}')
    print(f'Density: {delta_g0}')

    # Output folders
    base_dir = f'./data/{dataset_name}'
    nd_dir = os.path.join(base_dir, 'node-dropout')
    ed_dir = os.path.join(base_dir, 'edge-dropout')
    os.makedirs(nd_dir, exist_ok=True)
    os.makedirs(ed_dir, exist_ok=True)

    print('\nSTART GRAPH SAMPLING...')

    # --- Statistics file handling: append if resuming from a middle index ---
    stats_path = os.path.join(base_dir, 'sampling-stats.tsv')
    file_mode = 'w' if START_IDX == 0 else 'a'
    write_header = (file_mode == 'w') or (not os.path.exists(stats_path))

    with open(stats_path, file_mode) as f:
        fieldnames = ['dataset_id', 'strategy', 'dropout', 'users', 'items', 'interactions', 'delta_g']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        if write_header:
            writer.writeheader()

        # Nothing to do if START_IDX >= NUM_SAMPLINGS
        if START_IDX >= NUM_SAMPLINGS:
            print(f"Niente da generare: START_IDX ({START_IDX}) >= NUM_SAMPLINGS ({NUM_SAMPLINGS})")
            return

        # Use START_IDX in the range [START_IDX, NUM_SAMPLINGS-1]
        for idx in range(START_IDX, NUM_SAMPLINGS):
            # Deterministic seed per idx => reproducibility across machines
            set_all_seeds(RANDOM_SEED + idx)

            gss = random.choice(SAMPLING_STRATEGIES)
            dr = float(np.random.uniform(0.7, 0.9))

            # Utility function to save the two output formats
            def save_two_versions(current_dir, sampled_df, sample_idx):
                # 1) "idx.tsv" version without header: user_id, item_id, 1
                lite = sampled_df[[user_col, item_col]].copy()
                lite['__one__'] = 1
                lite.to_csv(os.path.join(current_dir, f'{sample_idx}.tsv'),
                            sep='\t', header=False, index=False)

                # 2) "header_idx.tsv" version with header and all original columns
                ordered_cols = []
                for c in [user_col, item_col, rating_col, ts_col]:
                    if c in sampled_df.columns:
                        ordered_cols.append(c)
                for c in sampled_df.columns:
                    if c not in ordered_cols and c not in ['user_idx', 'item_idx']:
                        ordered_cols.append(c)

                sampled_df[ordered_cols].to_csv(
                    os.path.join(current_dir, f'header_{sample_idx}.tsv'),
                    sep='\t', header=True, index=False
                )

            if gss == 'ND':
                print(f'\nRunning NODE DROPOUT (dataset={dataset_name}) with dropout ratio {dr}')
                sampled_edge_index, _, _ = dropout_node(edge_index, p=dr, num_nodes=num_users + num_items)

                stats, maybe_filtered = calculate_statistics_private(sampled_edge_index, num_users, num_items)
                if maybe_filtered is not None:
                    sampled_edge_index = maybe_filtered

                sampled_pairs = set(zip(sampled_edge_index[0].tolist(), sampled_edge_index[1].tolist()))
                sampled_df = dataset[dataset.apply(
                    lambda r: (int(r['user_idx']), int(r['item_idx'])) in sampled_pairs, axis=1
                )].copy()

                save_two_versions(nd_dir, sampled_df, idx)

                writer.writerow({
                    'dataset_id': idx,
                    'strategy': 'node dropout',
                    'dropout': dr,
                    **stats
                })

            elif gss == 'ED':
                print(f'\nRunning EDGE DROPOUT (dataset={dataset_name}) with dropout ratio {dr}')
                sampled_edge_index, _ = dropout_edge(edge_index, p=dr)

                stats, maybe_filtered = calculate_statistics_private(sampled_edge_index, num_users, num_items)
                if maybe_filtered is not None:
                    sampled_edge_index = maybe_filtered

                sampled_pairs = set(zip(sampled_edge_index[0].tolist(), sampled_edge_index[1].tolist()))
                sampled_df = dataset[dataset.apply(
                    lambda r: (int(r['user_idx']), int(r['item_idx'])) in sampled_pairs, axis=1
                )].copy()

                save_two_versions(ed_dir, sampled_df, idx)

                writer.writerow({
                    'dataset_id': idx,
                    'strategy': 'edge dropout',
                    'dropout': dr,
                    **stats
                })
            else:
                raise NotImplementedError('This graph sampling strategy has not been implemented yet!')


if __name__ == '__main__':
    # Automatically run sampling on all default datasets
    for ds in DATASETS:
        graph_sampling(ds)