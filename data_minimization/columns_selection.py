import pandas as pd

"""
The function in this file is used to flexibly select specific columns from a DataFrame based on keywords 
present in the column names. This is particularly useful when working with datasets that have inconsistent 
column naming conventions, such as varying cases (e.g., 'user_id' vs 'USER_ID'), additional suffixes 
(e.g., 'user_id:token'), or entirely different formats.

The function identifies and selects columns by searching for specified keywords within the column names. 
It also renames the selected columns to standardized names for downstream processing.

Args:
    df (pd.DataFrame): The input DataFrame with inconsistent column names.
    keywords (dict): A dictionary mapping standardized column names to the identifying keywords to look 
                     for in the column names. For example:
                     {'user': 'user', 'item': 'item', 'rating': 'rating', 'timestamp': 'timestamp'}.

Returns:
    pd.DataFrame: A DataFrame containing the selected columns, renamed to the standardized names.

Raises:
    ValueError: If no column matches a keyword or if multiple columns match the same keyword.

Example:
    # Sample DataFrame
    data = {
        'USER_ID:token': [1, 2, 3],
        'ITEM_ID:TOKEN': [10, 20, 30],
        'RATING:float': [5.0, 4.5, 4.0],
        'TIMESTAMP:float': [1633024000, 1633025000, 1633026000]
    }
    df = pd.DataFrame(data)

    # Define keywords
    keywords = {'user': 'user', 'item': 'item', 'rating': 'rating', 'timestamp': 'timestamp'}

    # Select and rename columns
    selected_df = select_columns_by_keyword(df, keywords)
    print(selected_df)
"""
def select_columns_by_keyword(df, keywords):
    """
    Select specific columns from a DataFrame based on keywords in column names.

    Args:
        df (pd.DataFrame): The input DataFrame.
        keywords (dict): A mapping of desired column names to identifying keywords, e.g.,
                         {'user': 'user', 'item': 'item', 'rating': 'rating', 'timestamp': 'timestamp'}.

    Returns:
        pd.DataFrame: A DataFrame with selected columns renamed to the desired names.
    """
    # Normalize column names: convert to lowercase for case-insensitive matching
    normalized_columns = {col.lower(): col for col in df.columns}

    # Identify and select columns by keywords
    selected_columns = {}
    for desired_name, keyword in keywords.items():
        # Find matching column
        matching_columns = [
            original_name
            for normalized_name, original_name in normalized_columns.items()
            if keyword.lower() in normalized_name
        ]
        if len(matching_columns) == 0:
            raise ValueError(f"Column with keyword '{keyword}' not found.")
        elif len(matching_columns) > 1:
            raise ValueError(f"Multiple columns match the keyword '{keyword}': {matching_columns}")
        selected_columns[desired_name] = matching_columns[0]

    # Create a new DataFrame with selected and renamed columns
    return df[list(selected_columns.values())].rename(columns=selected_columns)


def run_example_method_1():
    # Define the keywords for each desired column
    keywords = {'user': 'user', 'item': 'item', 'rating': 'rating', 'timestamp': 'timestamp'}


    data = {
        'USER_ID:token': [1, 2, 3],
        'ITEM_ID:TOKEN': [10, 20, 30],
        'RATING:float': [5.0, 4.5, 4.0],
        'TIMESTAMP:float': [1633024000, 1633025000, 1633026000]
    }
    df = pd.DataFrame(data)
    print(df)
    print(f"\n keywords: {keywords}")


    # Select and rename columns flexibly
    selected_df = select_columns_by_keyword(df, keywords)
    print(f"\n {selected_df}")
