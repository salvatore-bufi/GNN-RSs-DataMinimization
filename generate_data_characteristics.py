import multiprocessing
import re
import tqdm
import argparse
import pandas as pd
from multiprocessing import Pool
# multiprocessing.set_start_method('fork')
import os
import numpy as np
from operator import itemgetter

from data_characteristic import GraphDataset

# ==========================
# COMMON SETTINGS
# ==========================
DATASET = ['amazon-book', 'yelp']

AMAZON_COLUMNS_NAME = {
    'user_col': 'user_id',
    'item_col': 'parent_asin',
    'rating_col': 'rating',
    'timestamp_col': 'timestamp'
}

YELP_COLUMNS_NAME = {
    'user_col': 'user_id',
    'item_col': 'business_id',
    'rating_col': 'stars',
    'timestamp_col': 'date'
}

DATASET_COLUMNS_NAME = {
    'yelp': YELP_COLUMNS_NAME,
    'amazon-book': AMAZON_COLUMNS_NAME
}

# ==========================
# Minimized Dataset - Setting
# ==========================
DATA_DIRECTORY_STAT = './dataset'
INTERACTIONS_N = [1, 3, 6, 9, 12, 15, 18, 50, 100]
# full is not included in this list
STRATEGIES_STAT = [
    'random', 'most_recent', 'most_favorite',
    'least_favorite', 'most_rated', 'most_characteristic',
    'highest_variance'
]
LOG_ERROR_PATH_STAT = os.path.abspath('./log_error_characteristics.txt')
OUTPUT_STAT_PATH = './Dataset_Characteristics.tsv'

# ==========================
# Sampled Dataset - Setting
# ==========================
DATA_DIRECTORY_SAMPLES = './data'
SAMPLES_NUMBER = [i for i in range(1000)]   # 0..999
STRATEGIES_SAMPLES = ['edge-dropout', 'node-dropout']
LOG_ERROR_PATH_SAMPLES = os.path.abspath('./log_error_characteristics_samples.txt')
OUTPUT_SAMPLES_PATH = './Dataset_Characteristics_samples.tsv'


# ==========================
# COMMON FUNCTIONS
# ==========================
def compute_characteristics_on_dataset(
    dataset: pd.DataFrame,
    dataset_name=None,
    columns_names=YELP_COLUMNS_NAME,
    cutoff: int = 0,
    strategy: str = 'None'
):
    """
    Wrap around GraphDataset to compute all characteristics and return
    the resulting DataFrame.
    """
    gd = GraphDataset(dataset)
    gd.set_columns(
        user_col=columns_names['user_col'],
        item_col=columns_names['item_col'],
        rating_col=columns_names['rating_col'],
        timestamp_col=columns_names['timestamp_col']
    )
    characteristics_df = gd.compute_all_characteristics(
        dataset_name=dataset_name,
        strategy=strategy,
        cutoff=cutoff
    )
    return characteristics_df


# ==========================
# SCRIPT - Minimized Dataset
# ==========================
def load_datasets_min(dataset: str, strategy: str, cutoff: int, timestamp_col=None):
    """
    It reads files named: stat_<cutoff>.tsv
    from: ./dataset/<dataset>/<strategy>/
    """
    dataset_name_cutoff = 'stat' + '_' + str(cutoff) + '.tsv'
    dataset_path = os.path.abspath(os.path.join(DATA_DIRECTORY_STAT, dataset, strategy, dataset_name_cutoff))
    if timestamp_col is not None:
        try:
            if dataset == 'yelp':
                df = pd.read_csv(dataset_path, sep='\t', parse_dates=[timestamp_col])
                if not np.issubdtype(df[timestamp_col].dtype, np.number):
                    # If datetime, convert to Unix timestamp (seconds)
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col]).astype(np.int64) // 10 ** 9
                elif df[timestamp_col].max() > 1e12:
                    # If already numeric but in ms, convert to seconds
                    df[timestamp_col] = (df[timestamp_col] // 1000).astype(int)
            elif dataset == 'amazon-book':
                df = pd.read_csv(dataset_path, sep='\t')
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            with open(LOG_ERROR_PATH_STAT, 'a') as log_error_file:
                error_msg = (
                    f'Error Processing: \t {dataset} \t {strategy} \t '
                    f'{cutoff} : \t{str(e)} \n \n '
                )
                log_error_file.write(error_msg)
                return None
    else:
        try:
            df = pd.read_csv(dataset_path, sep='\t')
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            with open(LOG_ERROR_PATH_STAT, 'a') as log_error_file:
                error_msg = (
                    f'Error Processing: \t {dataset} \t {strategy} \t '
                    f'{cutoff} : \t{str(e)} \n \n '
                )
                log_error_file.write(error_msg)
                return None


def run_min_characteristics():
    """
    Reproduce exactly the behavior of Script 1:
    - compute characteristics for stat_<cutoff>.tsv
    - save Dataset_Characteristics.tsv
    """
    results_characteristics = []

    # All except full
    for dataset in DATASET:
        try:
            for strategy in STRATEGIES_STAT:
                for interactions_number in INTERACTIONS_N:
                    print(f"Computing (STAT) for {dataset}, {strategy}, {interactions_number}")
                    df = load_datasets_min(
                        dataset=dataset,
                        strategy=strategy,
                        cutoff=interactions_number,
                        timestamp_col=DATASET_COLUMNS_NAME[dataset]['timestamp_col']
                    )
                    # Script 1 did not check df is None; we preserve behavior.
                    df_characteristic = compute_characteristics_on_dataset(
                        dataset=df,
                        dataset_name=dataset,
                        columns_names=DATASET_COLUMNS_NAME[dataset],
                        cutoff=interactions_number,
                        strategy=strategy
                    )
                    if df_characteristic is not None and not df_characteristic.empty:
                        results_characteristics.append(df_characteristic)
        except Exception as e:
            with open(LOG_ERROR_PATH_STAT, 'a') as log_error_file:
                error_msg = (
                    f'Error Processing: \t {dataset} \t {strategy} \t '
                    f'{interactions_number} : \t{str(e)} \n \n '
                )
                log_error_file.write(error_msg)

    # ONLY FULL
    for dataset in DATASET:
        try:
            df = load_datasets_min(
                dataset=dataset,
                strategy='full',
                cutoff=1,
                timestamp_col=DATASET_COLUMNS_NAME[dataset]['timestamp_col']
            )
            df_characteristic = compute_characteristics_on_dataset(
                dataset=df,
                dataset_name=dataset,
                columns_names=DATASET_COLUMNS_NAME[dataset],
                cutoff=1,
                strategy='full'
            )
            if df_characteristic is not None and not df_characteristic.empty:
                results_characteristics.append(df_characteristic)
        except Exception as e:
            with open(LOG_ERROR_PATH_STAT, 'a') as log_error_file:
                error_msg = (
                    f'Error Processing: \t {dataset} \t FULL \t '
                    f'{1} : \t{str(e)} \n \n '
                )
                log_error_file.write(error_msg)

    # Final results (Script 1 always concatenated without checking empty)
    results = pd.concat(results_characteristics, ignore_index=True)
    results.to_csv(OUTPUT_STAT_PATH, sep='\t', index=False)



def load_datasets_samples(dataset: str, strategy: str, cutoff: int, timestamp_col=None):
    """
    Original load_datasets from Script 2.

    It reads files named: header_<cutoff>.tsv
    from: ./data/<dataset>/<strategy>/
    """
    dataset_name_cutoff = 'header' + '_' + str(cutoff) + '.tsv'
    dataset_path = os.path.abspath(os.path.join(DATA_DIRECTORY_SAMPLES, dataset, strategy, dataset_name_cutoff))
    if timestamp_col is not None:
        try:
            if dataset == 'yelp':
                df = pd.read_csv(dataset_path, sep='\t', parse_dates=[timestamp_col])
                if not np.issubdtype(df[timestamp_col].dtype, np.number):
                    # If datetime, convert to Unix timestamp (seconds)
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col]).astype(np.int64) // 10 ** 9
                elif df[timestamp_col].max() > 1e12:
                    # If already numeric but in ms, convert to seconds
                    df[timestamp_col] = (df[timestamp_col] // 1000).astype(int)
            elif dataset == 'amazon-book':
                df = pd.read_csv(dataset_path, sep='\t')
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            with open(LOG_ERROR_PATH_SAMPLES, 'a') as log_error_file:
                error_msg = (
                    f'Error Processing: \t {dataset} \t {strategy} \t '
                    f'{cutoff} : \t{str(e)} \n \n '
                )
                log_error_file.write(error_msg)
                return None
    else:
        try:
            df = pd.read_csv(dataset_path, sep='\t')
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
            with open(LOG_ERROR_PATH_SAMPLES, 'a') as log_error_file:
                error_msg = (
                    f'Error Processing: \t {dataset} \t {strategy} \t '
                    f'{cutoff} : \t{str(e)} \n \n '
                )
                log_error_file.write(error_msg)
                return None


def run_samples_characteristics():
    """
    Reproduce exactly the behavior of Script 2:
    - compute characteristics for header_<cutoff>.tsv
    - save Dataset_Characteristics_samples.tsv
    """
    results_characteristics = []

    for dataset in DATASET:
        for strategy in STRATEGIES_SAMPLES:
            for interactions_number in SAMPLES_NUMBER:
                try:
                    df = load_datasets_samples(
                        dataset=dataset,
                        strategy=strategy,
                        cutoff=interactions_number,
                        timestamp_col=DATASET_COLUMNS_NAME[dataset]['timestamp_col']
                    )
                    if df is None:
                        continue

                    df_characteristic = compute_characteristics_on_dataset(
                        dataset=df,
                        dataset_name=dataset,
                        columns_names=DATASET_COLUMNS_NAME[dataset],
                        cutoff=interactions_number,
                        strategy=strategy
                    )

                    if df_characteristic is not None and not df_characteristic.empty:
                        results_characteristics.append(df_characteristic)

                except Exception as e:
                    with open(LOG_ERROR_PATH_SAMPLES, 'a') as log_error_file:
                        error_msg = (
                            f'Error Processing: \t {dataset} \t {strategy} \t '
                            f'{interactions_number} : \t{str(e)} \n \n '
                        )
                        log_error_file.write(error_msg)

    # Final results (Script 2 checks if the list is empty)
    if results_characteristics:
        results = pd.concat(results_characteristics, ignore_index=True)
        results.to_csv(OUTPUT_SAMPLES_PATH, sep='\t', index=False)
    else:
        with open(LOG_ERROR_PATH_SAMPLES, 'a') as log_error_file:
            log_error_file.write('No characteristics computed; nothing to save.\n')


# ==========================
# MAIN: run both pipelines
# ==========================
if __name__ == '__main__':
    # First: original Script 1 behavior
    run_min_characteristics()

    # Then: original Script 2 behavior
    run_samples_characteristics()