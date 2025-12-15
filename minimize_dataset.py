import os
import pandas as pd
from typing import List
import shutil

from data_minimization import data_splitting
from data_minimization import minimization_strategies
from minimize_functions import apply_minimization_strategy

# Number of interactions to keep per user
INTERACTIONS_N = [1, 3, 6, 9, 12, 15, 18, 50, 100]

# All minimization strategies (full is handled separately at the end)
STRATEGIES = [
    'full',
    'random',
    'most_recent',
    'most_favorite',
    'least_favorite',
    'most_rated',
    'most_characteristic',
    'highest_variance',
]

DATA_PATH = os.path.abspath('./data')

# Dataset-specific column names
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


def run_minimization_for_dataset(dataset_name: str, columns: dict):
    data_path = os.path.join(DATA_PATH, dataset_name)
    candidate_path = os.path.join(data_path, 'dm_candidate.tsv')

    # Parse dates using the dataset-specific timestamp column
    dataframe = pd.read_csv(
        candidate_path,
        sep='\t',
        parse_dates=[columns['timestamp_col']]
    )

    print(f"Applying Minimization Strategies: {dataset_name} Dataset")

    # All strategies except 'full' for each n in INTERACTIONS_N
    for n in INTERACTIONS_N:
        print(f"Minimizing {n} interactions")


        for strategy in STRATEGIES:
            if strategy == 'full':
                continue

            print(f"\t\t Strategy: {strategy}")

            # Common arguments
            kwargs = dict(
                df=dataframe,
                dataset=dataset_name,
                strategy=strategy,
                n=n,
                user_col_name=columns['user_col'],
                item_col_name=columns['item_col'],
                rating_col_name=columns['rating_col'],
            )

            # Only 'most_recent' needs the timestamp column
            if strategy == 'most_recent':
                kwargs['timestamp_col_name'] = columns['timestamp_col']

            apply_minimization_strategy(**kwargs)

        print("\n")

    # 'full' strategy called once with n=1 for each dataset
    print(f"Applying FULL strategy for {dataset_name} (n=1)")
    apply_minimization_strategy(
        df=dataframe,
        dataset=dataset_name,
        strategy='full',
        n=1,
        user_col_name=columns['user_col'],
        item_col_name=columns['item_col'],
        rating_col_name=columns['rating_col'],
    )


if __name__ == '__main__':
    # Run the same pipeline for both amazon-book and yelp
    for dataset_name, columns in DATASET_SCHEMAS.items():
        run_minimization_for_dataset(dataset_name, columns)