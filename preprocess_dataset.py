# https://business.yelp.com/data/resources/open-dataset/
import os
import json
import random
import numpy as np
import pandas as pd

from data_minimization import data_splitting

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

DATA_PATH = os.path.abspath('./data')

# Dataset-specific configuration
DATASET_CONFIG = {
    'amazon-book': {
        'pretty_name': 'AMAZON',
        'data_dir': os.path.join(DATA_PATH, 'amazon-book'),
        'input_path': os.path.join(DATA_PATH, 'amazon-book', 'Books.csv.gz'),
        'k_core': 20,
        'columns': {
            'user_col': 'user_id',
            'item_col': 'parent_asin',
            'rating_col': 'rating',
            'timestamp_col': 'timestamp',
        },
        'type': 'amazon_csv',  # CSV file
    },
    'yelp': {
        'pretty_name': 'YELP',
        'data_dir': os.path.join(DATA_PATH, 'yelp'),
        'json_dir': os.path.join(DATA_PATH, 'yelp', 'Yelp JSON'),
        'input_path': os.path.join(DATA_PATH, 'yelp', 'Yelp JSON', 'yelp_academic_dataset_review.json'),
        'k_core': 20,
        'columns': {
            'user_col': 'user_id',
            'item_col': 'business_id',
            'rating_col': 'stars',
            'timestamp_col': 'date',
        },
        'type': 'yelp_json',  # JSON lines file
    },
}


def read_ratings_to_df(filepath: str, columns: dict) -> pd.DataFrame:
    """
    Read the Yelp JSON lines file and keep only the required fields.
    This reproduces your original read_ratings_to_df.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            filtered_record = {
                "user_id": record[columns['user_col']],
                "business_id": record[columns['item_col']],
                "stars": record[columns['rating_col']],
                "date": record[columns['timestamp_col']],
            }
            data.append(filtered_record)

    df = pd.DataFrame(data, columns=["user_id", "business_id", "stars", "date"])
    return df


def read_dataset(dataset_name: str, cfg: dict) -> pd.DataFrame:
    """
    Read the raw dataset using the same logic as the two original scripts.
    """
    input_path = cfg['input_path']
    pretty_name = cfg['pretty_name']
    columns = cfg['columns']

    print(f" Preprocessing Dataset: {pretty_name}")

    if not os.path.isfile(input_path):
        if dataset_name == 'amazon-book':
            raise FileNotFoundError(
                f"The dataset file 'Books.csv.gz' was not found at the path: {input_path} \n"
            )
        elif dataset_name == 'yelp':
            json_dir = cfg['json_dir']
            data_dir = cfg['data_dir']
            raise FileNotFoundError(
                f"The dataset file 'yelp_academic_dataset_review.json' was not found at the path: {json_dir} \n"
                f"Please ensure that you have extracted the file 'Yelp-JSON.zip' in the directory {data_dir}"
            )

    if cfg['type'] == 'amazon_csv':
        df = pd.read_csv(input_path)
    elif cfg['type'] == 'yelp_json':
        df = read_ratings_to_df(filepath=input_path, columns=columns)
    else:
        raise ValueError(f"Unknown dataset type: {cfg['type']}")

    return df


def process_dataset(dataset_name: str, cfg: dict):
    """
    Full preprocessing pipeline for a dataset:
    - read raw data
    - compute statistics
    - apply iterative k-core
    - split into candidate / val / test
    - save all TSV files in the same format as the original scripts
    """
    columns = cfg['columns']
    data_dir = cfg['data_dir']
    k_core = cfg['k_core']

    # 1) Read dataframe
    df = read_dataset(dataset_name, cfg)
    print("Original DF \t - Dataset Statistic")
    data_splitting.dataset_statistic(df, columns)

    # 2) Apply iterative k-core
    print(f"Applying Iterative K-core: {k_core}-core")
    df_cored = data_splitting.iterative_k_core(
        df,
        column1=columns['user_col'],
        column2=columns['item_col'],
        k=k_core
    )
    print(f"{k_core}-cored DF")

    print(f"{k_core}-cored DF \t - Dataset Statistic")
    data_splitting.dataset_statistic(df_cored, columns)

    # 3) Split into candidate / val / test
    print(
        f"Splitting the Dataset - {TRAIN_RATIO * 100} Candidate \t "
        f"{VAL_RATIO * 100}  Validation \t {TEST_RATIO * 100} Test"
    )
    candidate, val, test = data_splitting.split_dataset_per_user(
        df_cored,
        columns['user_col'],
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

    # 4) Save to disk (same as in the original scripts)
    print(f"Saving Splitted Dataset in {data_dir}")
    candidate_path = os.path.join(data_dir, 'dm_candidate.tsv')
    full_train_path = os.path.join(data_dir, 'train.tsv')      # candidate, ready for Elliot
    val_path = os.path.join(data_dir, 'val_full.tsv')
    val_elliot_path = os.path.join(data_dir, 'val.tsv')
    test_path = os.path.join(data_dir, 'test_full.tsv')
    test_elliot_path = os.path.join(data_dir, 'test.tsv')

    # Candidate (train)
    candidate.to_csv(candidate_path, sep='\t', index=False)

    # Train (Elliot format: rating set to 1, no header)
    full_train_elliot = candidate[
        [columns['user_col'], columns['item_col'], columns['rating_col']]
    ].copy()
    full_train_elliot[columns['rating_col']] = 1
    full_train_elliot.to_csv(full_train_path, sep='\t', index=False, header=False)

    # Validation
    val.to_csv(val_path, sep='\t', index=False)
    val_elliot = val[
        [columns['user_col'], columns['item_col'], columns['rating_col']]
    ].copy()
    val_elliot[columns['rating_col']] = 1
    val_elliot.to_csv(val_elliot_path, sep='\t', index=False, header=False)

    # Test
    test.to_csv(test_path, sep='\t', index=False)
    test_elliot = test[
        [columns['user_col'], columns['item_col'], columns['rating_col']]
    ].copy()
    test_elliot[columns['rating_col']] = 1
    test_elliot.to_csv(test_elliot_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    # Process both datasets: amazon-book and yelp
    for dataset_name, cfg in DATASET_CONFIG.items():
        process_dataset(dataset_name, cfg)