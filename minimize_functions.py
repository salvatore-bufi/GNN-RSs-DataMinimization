import os
import pandas as pd
from typing import List
import shutil

from data_minimization import data_splitting
from data_minimization import minimization_strategies

INTERACTIONS_N = [1, 3, 7, 15, 100]

STRATEGIES = ['full', 'random', 'most_recent', 'most_favorite', 'least_favorite', 'most_rated', 'most_characteristic',
              'highest_variance']

function_mapping = {
    'full': minimization_strategies.full_min,
    'random': minimization_strategies.random_min,
    'most_recent': minimization_strategies.most_recent_min,
    'most_favorite': minimization_strategies.most_favorite_min,
    'least_favorite': minimization_strategies.least_favorite_min,
    'most_rated': minimization_strategies.most_rated_min,
    'most_characteristic': minimization_strategies.most_characteristic_min,
    'highest_variance': minimization_strategies.highest_variance_min
}


def check_k_core(df: pd.DataFrame, column_name: str = 'user_id:token'):
    val_count = df[column_name].value_counts()
    min_count = val_count.min()
    min_value = val_count.idxmin()
    print(min_count, min_value)


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created : {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    return None


def copy_and_rename(source_file_path: str, dest_file_path: str) -> None:
    if not os.path.isfile(source_file_path):
        raise FileNotFoundError(f" File {source_file_path} not found !")
    try:
        shutil.copy2(source_file_path, dest_file_path)
        print(f" Copied {source_file_path} \t {dest_file_path}")
    except IOError as e:
        print(f" Unable to copy file. {e}")
    except Exception as e:
        print(f"Unexpected error {e}")


def apply_minimization_strategy(df: pd.DataFrame,  dataset: str = 'ml-1m',
                                strategy: str = 'full', val_path: str = None, test_path: str = None, **kwargs) -> None:
    '''

    Args:
        df: candidate dataframe ( DM_candidate)
        df_path: directory containing all splitting ( DM_candidate, test, val )
        dataset: name of the dataset
        strategy: name of minimization strategy to apply
        **kwargs:

    Returns:

    '''
    strategy = strategy.lower()
    if strategy not in STRATEGIES:
        all_strategies_names = "\n".join(["\t-" + n for n in STRATEGIES])
        raise ValueError(
            f" Strategy: {strategy} not implemented.\n The implemented strategies are:\n {all_strategies_names}")

    # df = pd.read_csv(os.path.join(df_path, df_name), sep='\t')
    try:
        func = function_mapping[strategy]
        minimized_df = func(df, **kwargs)
    except TypeError as e:
        raise TypeError(f" Error calling function '{strategy}': {e}")

    # new_df_name = dataset + '-' + strategy
    new_df_path_method = os.path.abspath(os.path.join('./dataset', dataset, strategy))
    create_directory(new_df_path_method)  # directory for the method minimization

    train_file_name = f"{kwargs['n']}" + '.tsv'
    # train_file_name = f"{n}" + '.tsv'

    # minimized_df.to_csv(os.path.join(new_df_path_method, train_file_name + 'headers_full'), sep='\t', index=False)

    user_col_name = kwargs['user_col_name']
    item_col_name = kwargs['item_col_name']
    rating_col_name = kwargs['rating_col_name']

    minimized_df_elliot = minimized_df[[user_col_name, item_col_name, rating_col_name]].copy()
    minimized_df_elliot[rating_col_name] = 1
    minimized_df_elliot.to_csv(os.path.join(new_df_path_method, train_file_name), sep='\t', index=False, header=False)

    # path = df_path
    #
    # dest_val_path = os.path.abspath(os.path.join('./dataset', dataset, 'val.tsv'))
    # if not os.path.isfile(dest_val_path):
    #     if val_path is None:
    #         val_path = os.path.join(path, 'dm_val.tsv')  # actual
    #         copy_and_rename(val_path, dest_val_path)
    #
    # dest_test_path = os.path.abspath(os.path.join('./dataset', dataset, 'test.tsv'))
    # if test_path is None:
    #     test_path = os.path.join(path, 'dm_test.tsv')  # actual
    #     copy_and_rename(test_path, dest_test_path)
    return

def apply_minimization_strategy_fullcolumns(df: pd.DataFrame,  dataset: str = 'ml-1m',
                                strategy: str = 'full', val_path: str = None, test_path: str = None, **kwargs) -> None:
    '''

    Args:
        df: candidate dataframe ( DM_candidate)
        df_path: directory containing all splitting ( DM_candidate, test, val )
        dataset: name of the dataset
        strategy: name of minimization strategy to apply
        **kwargs:

    Returns:

    '''
    strategy = strategy.lower()
    if strategy not in STRATEGIES:
        all_strategies_names = "\n".join(["\t-" + n for n in STRATEGIES])
        raise ValueError(
            f" Strategy: {strategy} not implemented.\n The implemented strategies are:\n {all_strategies_names}")


    try:
        func = function_mapping[strategy]
        minimized_df = func(df, **kwargs)
    except TypeError as e:
        raise TypeError(f" Error calling function '{strategy}': {e}")


    new_df_path_method = os.path.abspath(os.path.join('./dataset', dataset, strategy))
    create_directory(new_df_path_method)  # directory for the method minimization

    train_file_name = f"stat_{kwargs['n']}" + '.tsv'

    minimized_df.to_csv(os.path.join(new_df_path_method, train_file_name), sep='\t', index=False, header=True)
    return


def apply_minimization_strategy_old(df: pd.DataFrame, df_path: str, dataset: str = 'ml-1m',
                                strategy: str = 'full', val_path: str = None, test_path: str = None, **kwargs) -> None:
    '''

    Args:
        df: candidate dataframe ( DM_candidate)
        df_path: directory containing all splitting ( DM_candidate, test, val )
        dataset: name of the dataset
        strategy: name of minimization strategy to apply
        val_path:
        test_path:
        **kwargs:

    Returns:

    '''
    strategy = strategy.lower()
    if strategy not in STRATEGIES:
        all_strategies_names = "\n".join(["\t-" + n for n in STRATEGIES])
        raise ValueError(
            f" Strategy: {strategy} not implemented.\n The implemented strategies are:\n {all_strategies_names}")

    # df = pd.read_csv(os.path.join(df_path, df_name), sep='\t')
    try:
        func = function_mapping[strategy]
        minimized_df = func(df, **kwargs)
    except TypeError as e:
        raise TypeError(f" Error calling function '{strategy}': {e}")

    # new_df_name = dataset + '-' + strategy
    new_df_path_method = os.path.abspath(os.path.join('./dataset', dataset, strategy))
    create_directory(new_df_path_method)  # directory for the method minimization

    train_file_name = f"{kwargs['n']}" + '.tsv'

    user_col_name = kwargs['user_col_name']
    item_col_name = kwargs['item_col_name']
    rating_col_name = kwargs['rating_col_name']

    minimized_df_elliot = minimized_df[[user_col_name, item_col_name, rating_col_name]].copy()
    minimized_df_elliot.to_csv(os.path.join(new_df_path_method, train_file_name), sep='\t', index=False, header=False)
    return
