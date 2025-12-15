import pandas as pd
import numpy as np

DEFAULT_SEED = 42
np.random.seed(DEFAULT_SEED)


############## The following strategies align with those presented in the paper "Operationalizing the Legal Principle of Data Minimization for Personalization", where n represents the number of selected items per user
# Full minimization (no minimization, return all data)
def full_min(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return df.copy()


# Random minimization
def random_min(df: pd.DataFrame, n: int, user_col_name: str = 'user_id:token', **kwargs) -> pd.DataFrame:
    return df.groupby(user_col_name).apply(lambda x: x.sample(n=min(len(x), n), random_state=42)).reset_index(drop=True)


# Most recent minimization
def most_recent_min(df: pd.DataFrame, n: int, user_col_name: str = 'user_id:token',
                    timestamp_col_name: str = 'timestamp:float', item_col_name: str = 'item_id:token',
                    **kwargs) -> pd.DataFrame:
    # Assuming the dataset has a 'timestamp' column for ordering by time
    if timestamp_col_name not in df.columns:
        raise ValueError("Most recent minimization requires a 'timestamp' column.")
    # item_col_name added for stability: case , same user equal timestamp
    return df.sort_values(by=[user_col_name, timestamp_col_name, item_col_name], ascending=[True, False, True]).groupby(
        user_col_name).head(n).reset_index(
        drop=True)


# Most favorite minimization
def most_favorite_min(df: pd.DataFrame, n: int, user_col_name: str = 'user_id:token',
                      rating_col_name: str = 'rating:float', item_col_name: str = 'item_id:token',
                      **kwargs) -> pd.DataFrame:
    # Sort by rating (desc) and then item_id (asc) as a tie-breaker
    return df.sort_values(by=[user_col_name, rating_col_name, item_col_name], ascending=[True, False, True]).groupby(
        user_col_name).head(n).reset_index(drop=True)
    # return df.groupby(user_col_name).apply(lambda x: x.nlargest(n=min(len(x), n), columns=rating_col_name)).reset_index(drop=True)


# Least favorite minimization
def least_favorite_min(df: pd.DataFrame, n: int, user_col_name: str = 'user_id:token',
                       rating_col_name: str = 'rating:float', item_col_name: str = 'item_id:token',
                       **kwargs) -> pd.DataFrame:
    return df.sort_values(by=[user_col_name, rating_col_name, item_col_name], ascending=[True, True, True]).groupby(
        user_col_name).head(n).reset_index(drop=True)
    # return df.groupby(user_col_name).apply(lambda x: x.nsmallest(n=min(len(x), n), columns=rating_col_name)).reset_index(drop=True)


# Most rated minimization
# def most_rated_min_old(df: pd.DataFrame, n: int, item_col_name: str = 'item_id:token', user_col_name: str = 'user_id:token',
#                    **kwargs) -> pd.DataFrame:
#     item_counts = df[item_col_name].value_counts().to_dict()
#     df['item_rating_count'] = df[item_col_name].map(item_counts)
#     minimized = df.sort_values(by=[user_col_name, 'item_rating_count'], ascending=[True, False]).groupby(
#         user_col_name).head(
#         n).reset_index(drop=True)
#     df.drop(columns='item_rating_count', inplace=True)  # Cleanup temporary column
#     return minimized

def most_rated_min(df: pd.DataFrame, n: int,
                   item_col_name: str = 'item_id:token',
                   user_col_name: str = 'user_id:token',
                   **kwargs) -> pd.DataFrame:
    df_copy = df.copy() # Work on a copy to avoid modifying the original df
    item_counts = df_copy[item_col_name].value_counts().to_dict()
    df_copy['item_rating_count'] = df_copy[item_col_name].map(item_counts)
    minimized = (df_copy.sort_values(by=[user_col_name, 'item_rating_count', item_col_name], ascending=[True, False, True])
                        .groupby(user_col_name)
                        .head(n)
                        .reset_index(drop=True)
                        .drop(columns='item_rating_count')) # Drop the column from the final result
    return minimized


# Most characteristic minimization
def most_characteristic_min_old(df: pd.DataFrame, n: int, item_col_name: str = 'item_id:token',
                                user_col_name: str = 'user_id:token', **kwargs) -> pd.DataFrame:
    # Construct a characteristic score based on user-item interactions
    item_characteristics = df.groupby(item_col_name).apply(lambda x: len(x[user_col_name].unique())).to_dict()
    df['item_characteristic_score'] = df[item_col_name].map(item_characteristics)
    minimized = df.sort_values(by=[user_col_name, 'item_characteristic_score'], ascending=[True, False]).groupby(
        user_col_name).head(n).reset_index(drop=True)
    df.drop(columns='item_characteristic_score', inplace=True)  # Cleanup temporary column
    return minimized


def most_characteristic_min(df: pd.DataFrame, n: int,
                            item_col_name: str = 'item_id:token',
                            user_col_name: str = 'user_id:token',
                            **kwargs) -> pd.DataFrame:
    # Step 1: Create a binary user-item interaction matrix
    user_item_matrix = pd.crosstab(df[item_col_name], df[user_col_name])

    # Step 2: Compute the system-wide average item profile
    avg_vector = user_item_matrix.mean(axis=0).to_frame().T

    # Step 3: Calculate Euclidean distance between each item's binary vector and the average vector
    item_vectors = user_item_matrix.values  # Binary vectors for all items
    # avg_vector = avg_vector.values.reshape(-1, 1)  # System-wide average as a column vector
    distances = np.linalg.norm(item_vectors - avg_vector.to_numpy(), axis=1)

    # Step 4: Map distances back to the item IDs
    item_distance_df = pd.DataFrame({
        item_col_name: user_item_matrix.index,
        'distance_to_avg': distances
    })

    # Step 5: Merge distances with the original DataFrame
    df = df.merge(item_distance_df, on=item_col_name)

    # Step 6: Sort by user and distance, then select top `n` items for each user
    minimized = (
        df.sort_values(by=[user_col_name, 'distance_to_avg', item_col_name], ascending=[True, True, True])
        .groupby(user_col_name)
        .head(n)
        .reset_index(drop=True)
    )

    # Return the minimized DataFrame
    return minimized


# Highest variance minimization
def highest_variance_min(df: pd.DataFrame, n: int, item_col_name: str = 'item_id:token',
                         user_col_name: str = 'user_id:token', rating_col_name: str = 'rating:float',
                         **kwargs) -> pd.DataFrame:
    item_variances = df.groupby(item_col_name)[rating_col_name].var().to_dict()
    df['item_variance'] = df[item_col_name].map(item_variances)
    minimized = df.sort_values(by=[user_col_name, 'item_variance', item_col_name], ascending=[True, False, True]).groupby(
        user_col_name).head(
        n).reset_index(drop=True)
    df.drop(columns='item_variance', inplace=True)  # Cleanup temporary column
    return minimized
