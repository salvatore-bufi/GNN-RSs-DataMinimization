import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import random


SEED = 42 # Or any integer
random.seed(SEED)
np.random.seed(SEED)


def dataset_statistic(df: pd.DataFrame, col_names: dict):
    # number of unique users
    n_user = df[col_names['user_col']].nunique()
    # number of unique items
    n_item = df[col_names['item_col']].nunique()
    # total observed interactions
    n_interactions = len(df)
    # density = observed / total possible
    density = n_interactions / (n_user * n_item)
    # sparsity = 1 - density
    sparsity = 1 - density
    # print out
    print(f"Number of users:   {n_user}")
    print(f"Number of items:   {n_item}")
    print(f"Total ratings:     {n_interactions}")
    print(f"Density:           {density:.6f}")
    print(f"Sparsity:          {sparsity:.6f}")

def subsample_by_column(df: pd.DataFrame, column: str, n:int = 2500, seed=42):
    """
    Subsamples N unique values from a specified column (by name or index) and extracts rows containing those values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str or int): The column to subsample unique values from (can be a column name or index).
    - n (int): The number of unique values to sample.

    Returns:
    - pd.DataFrame: A subset of the original DataFrame with rows matching the sampled values.
    """
    if seed is not None:
        random.seed(seed)
    # Get the column name if column is an index
    if isinstance(column, int):
        if column < 0 or column >= len(df.columns):
            raise ValueError(f"Column index {column} is out of bounds for the DataFrame.")
        column_name = df.columns[column]
    elif isinstance(column, str):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        column_name = column
    else:
        raise TypeError("Column must be either a string (column name) or an integer (column index).")

    # Get unique values from the column
    unique_values = df[column_name].unique()

    # Ensure we have enough unique values to sample
    if len(unique_values) < n:
        raise ValueError(
            f"Cannot sample {n} unique values from column '{column_name}' with only {len(unique_values)} unique values.")

    # Randomly sample N unique values
    sampled_values = random.sample(list(unique_values), n)

    # Filter the DataFrame to include only rows with the sampled values
    filtered_df = df[df[column_name].isin(sampled_values)]

    return filtered_df


def iterative_k_core(dataframe: pd.DataFrame, column1: str = 'user_id', column2: str = 'item_id', k: int = 5) -> pd.DataFrame:
    """
    Performs k-core filtering on the dataset based on two specified columns.
    Filters the dataset by deleting rows, such that each value in both specified columns
    in the final dataset has at least k interactions.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe.
    column1 (str): The first column name to perform k-core on. Default is 'user_id'.
    column2 (str): The second column name to perform k-core on. Default is 'item_id'.
    k (int): The minimum number of interactions required for both columns.

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    filtered_df = dataframe.copy()
    while True:
        # Count occurrences of each value in column1
        counts_col1 = filtered_df[column1].value_counts()
        to_remove_col1 = counts_col1[counts_col1 < k].index

        # Count occurrences of each value in column2
        counts_col2 = filtered_df[column2].value_counts()
        to_remove_col2 = counts_col2[counts_col2 < k].index

        # Check if any values need to be removed in either column
        if len(to_remove_col1) == 0 and len(to_remove_col2) == 0:
            break

        # Filter out rows with values in column1 that have counts less than k
        filtered_df = filtered_df[~filtered_df[column1].isin(to_remove_col1)]

        # Filter out rows with values in column2 that have counts less than k
        filtered_df = filtered_df[~filtered_df[column2].isin(to_remove_col2)]

    return filtered_df


def k_core(dataframe: pd.DataFrame, column: str = 'user_id', k: int = 5) -> pd.DataFrame:
    """
    Performs k-core filtering on the dataset based on the specified column.
    Filters the dataset by deleting rows, such that each value in the specified column in the final dataset has at least k interactions.

    Parameters:
    dataframe (pd.DataFrame): The input dataframe.
    column (str): The column name to perform k-core on. Default is 'user_id'.
    k (int): The minimum number of interactions required.

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    filtered_df = dataframe.copy()
    while True:
        # Count occurrences of each value in the specified column
        counts = filtered_df[column].value_counts()
        # Identify values with counts less than k
        to_remove = counts[counts < k].index
        # If there are no values to remove, break the loop
        if len(to_remove) == 0:
            break
        # Filter out rows with values that have counts less than k
        filtered_df = filtered_df[~filtered_df[column].isin(to_remove)]
    return filtered_df

def user_based_split(dataframe: pd.DataFrame, user_col: str = 'user_id', percentage: float = 0.5, seed: int = 42) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a dataset into two parts based on a percentage of unique users.

    Args:
        dataframe (pd.DataFrame): The input dataset.
        user_col (str): The name of the column representing the user IDs.
        percentage (float): The percentage (A) of users for the first split (0 < A < 1).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, the first with A% of users,
                                           and the second with (1-A)% of users.
    """
    # Ensure percentage is valid
    if not (0 < percentage < 1):
        raise ValueError("Percentage must be between 0 and 1.")

    # Extract unique users
    unique_users = dataframe[user_col].unique()
    total_users = len(unique_users)

    # Split unique users into two groups
    train_users, test_users = train_test_split(
        unique_users,
        train_size=percentage,
        random_state=seed,
        shuffle=True
    )

    # Create masks to filter the DataFrame for the user splits
    train_mask = dataframe[user_col].isin(train_users)
    test_mask = dataframe[user_col].isin(test_users)

    # Split the DataFrame based on the masks
    train_split = dataframe[train_mask]
    test_split = dataframe[test_mask]

    return train_split, test_split




def split_dataset_per_user_train_test(dataframe: pd.DataFrame, user_col: str = 'user_id', percentage: float = 0.5, seed: int = 42) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataset per user, distributing a percentage A of each user's interactions
    into the train set and (1-A) into the test set.

    Args:
        dataframe (pd.DataFrame): The dataset.
        user_col (str): The name of the column representing the user IDs.
        percentage (float): The percentage (A) of each user's interactions for the train split (0 < A < 1).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, the first containing A% of each user's interactions (train),
                                           and the second containing the rest (test).
    """
    # Ensure percentage is valid
    if not (0 < percentage < 1):
        raise ValueError("Percentage must be between 0 and 1.")

    train_splits = []
    test_splits = []

    # Group the dataset by user_id
    grouped = dataframe.groupby(user_col)

    for user_id, user_data in grouped:
        # Split each user's interactions
        train_data, test_data = train_test_split(
            user_data,
            train_size=percentage,
            random_state=seed,
            shuffle=True
        )
        train_splits.append(train_data)
        test_splits.append(test_data)

    # Concatenate all the user-specific splits into final DataFrames
    train_split = pd.concat(train_splits).reset_index(drop=True)
    test_split = pd.concat(test_splits).reset_index(drop=True)

    return train_split, test_split


def split_dataset_per_user(
    dataframe: pd.DataFrame,
    user_col: str = 'user_id',
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the dataset per user, distributing specified ratios of each user's interactions
    into the train, validation, and test sets.

    Args:
        dataframe (pd.DataFrame): The dataset.
        user_col (str): The name of the column representing the user IDs.
        train_ratio (float): The ratio of each user's interactions for the train split (0 < train_ratio < 1).
        val_ratio (float): The ratio of each user's interactions for the validation split (0 <= val_ratio < 1).
        test_ratio (float): The ratio of each user's interactions for the test split (0 < test_ratio < 1).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames containing the train, validation, and test splits.
    """
    # Ensure ratios are valid
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1) or not (0 < test_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum up to 1.")

    # Set a global random seed for reproducibility
    np.random.seed(seed)

    train_splits = []
    val_splits = []
    test_splits = []

    # Group the dataset by user_id
    grouped = dataframe.groupby(user_col)

    for user_id, user_data in grouped:
        # Shuffle user_data using the global random state
        user_data = user_data.sample(frac=1).reset_index(drop=True)

        n = len(user_data)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = n - n_train - n_val  # Ensure all data is assigned

        # Split the data
        train_data = user_data.iloc[:n_train]
        val_data = user_data.iloc[n_train:n_train + n_val]
        test_data = user_data.iloc[n_train + n_val:]

        train_splits.append(train_data)
        val_splits.append(val_data)
        test_splits.append(test_data)

    # Concatenate all the user-specific splits into final DataFrames
    train_split = pd.concat(train_splits).reset_index(drop=True)
    val_split = pd.concat(val_splits).reset_index(drop=True)
    test_split = pd.concat(test_splits).reset_index(drop=True)

    return train_split, val_split, test_split


def run_example_split() -> None:
    # Sample DataFrame
    data = {
        'user_id': [1, 1, 2, 2, 3, 3, 3, 4, 4],
        'item_id': [10, 20, 10, 30, 40, 50, 10, 50, 12],
        'rating': [5.0, 4.0, 4.5, 3.5, 5.0, 4.0, 3.0, 1.0, 4.0],
        'timestamp': [1633024000, 1633025000, 1633024000, 1633025000, 1633026000, 1633027000, 163302798, 1633027002, 1633027009]
    }
    df = pd.DataFrame(data)

    # k_cored = k_core(df, column='user_id', k=3)

    # User-based split
    train, test = user_based_split(df, user_col='user_id', percentage=0.7, seed=42)
    print("User-based split:")
    print("Train:")
    print(train)
    print("Test:")
    print(test)

    # Split dataset per user
    train, test, val = split_dataset_per_user(df, user_col='user_id', train_ratio=0.7, test_ratio=0.3, val_ratio=0.0, seed=42)
    print("\nSplit dataset per user:")
    print("Train:")
    print(train)
    print("Test:")
    print(test)


