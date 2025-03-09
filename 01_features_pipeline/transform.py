import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from typing import Dict, Any, Optional
import utils

logger = utils.get_logger(__name__)


def agg_user_movie_tags(
    df: pd.DataFrame
) -> pd.DataFrame:
    """ Aggregates user movie tags per movie
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to process

    Returns:
    --------
    pandas DataFrame
        The DataFrame with one row per user-movie with aggregated tags
    """
    df_return = (
        df.groupby(["userId", "movieId"])["tag"]
        .apply(", ".join)
        .reset_index()
        .rename(columns={"tag": "user_movie_tags"})
        .assign(user_movie_tags=lambda x: x["user_movie_tags"].str.lower())
    )
    return df_return


def handle_missing_values(
    df: pd.DataFrame, 
    column_rules: Optional[Dict[str, Any]] = None, 
    fill_numeric: float = None, 
    fill_categorical: str = None, 
    nan_threshold: float = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame with column-specific rules.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to process
    column_rules : dict, optional
        Dictionary with column names as keys and specific fill values or methods as values.
        Methods can be 'median', 'mean', 'mode', or any specific value.
        Can also be a function that takes a Series and returns a value.
        Example: {'rating': 'median', 'tags': 'mode', 'category': 'Unknown', 'views': lambda x: x.max() * 0.5}
    fill_numeric : float | int, optional
        Default fill value for numeric columns not in column_rules
    fill_categorical : str | List[str], optional
        Default fill value for categorical columns not in column_rules
    nan_threshold : float, optional
        Only process columns with NaN percentage above this threshold
    verbose : bool, optional
        Whether to print information about the filling process
        
    Returns:
    --------
    pandas DataFrame
        The DataFrame with missing values handled
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Calculate NaN percentages
    nan_percentage = df_processed.isna().mean() * 100
    nan_columns = nan_percentage[nan_percentage > 0]

    
    if not nan_columns.empty:
        logger.info("Columns with NaN values and their percentages:")
        logger.info(nan_columns.to_string())
    
    # If no column rules provided, initialize empty dict
    if column_rules is None:
        column_rules = {}
    
    # Define strategy functions for common methods
    strategies = {
        'median': lambda series: series.median(),
        'mean': lambda series: series.mean(),
        'mode': lambda series: series.mode()[0] if not series.mode().empty else None,
        'most_common': lambda series: series.mode()[0] if not series.mode().empty else None,
        'min': lambda series: series.min(),
        'max': lambda series: series.max(),
        'zero': lambda series: 0,
    }
    
    # Process columns based on NaN percentage
    for col, percent in nan_percentage.items():

        if percent == 0:
            print(f"No NaN values in column: {col}. Skipping to next column.")
            continue

        if percent < nan_threshold:
            logger.info(f"Dropping NaN subset in column: {col} (NaN {percent:.3f}%) because of NaN threshold {nan_threshold}")
            df_processed.dropna(subset=[col], inplace=True)
            continue
        
        # Skip columns that don't exist in the DataFrame
        if col not in df_processed.columns:
            continue
        
        # Get the column data for convenience
        col_data = df_processed[col]
        is_numeric = is_numeric_dtype(col_data)
        is_categorical = is_categorical_dtype(col_data) or col_data.dtype == 'object'
        
        # Determine the fill value
        fill_value = None
        
        # Check if column has specific rule
        if col in column_rules:
            rule = column_rules[col]
            
            # Rule is a string strategy
            if isinstance(rule, str) and rule.lower() in strategies:
                strategy = strategies[rule.lower()]
                # Check if strategy is applicable (e.g., mean/median for numeric only)
                if (rule.lower() in ['mean', 'median'] and not is_numeric):
                    logger.info(f"Warning: Cannot apply {rule} to non-numeric column {col}. Using mode instead.")
                    fill_value = strategies['mode'](col_data)
                else:
                    fill_value = strategy(col_data)
                
            # Rule is a function
            elif callable(rule):
                try:
                    fill_value = rule(col_data)
                except Exception as e:
                    logger.info(f"Error applying custom function to {col}: {e}")
                    # Fall back to default
                    fill_value = fill_numeric if is_numeric else fill_categorical
                    
            # Rule is a direct value
            else:
                fill_value = rule
        
        # Apply default handling based on column type if no rule or rule processing failed
        if fill_value is None:
            if is_numeric:
                if fill_numeric is None:
                    fill_value = col_data.median()
                else:
                    fill_value = fill_numeric
            elif is_categorical:
                if fill_categorical is None and not col_data.mode().empty:
                    fill_value = col_data.mode()[0]
                else:
                    fill_value = fill_categorical
        
        # Apply the fill value
        logger.info(f"Filling NaN in column: {col} (NaN {percent:.3f}%) with {fill_value}")
        
        df_processed[col].fillna(value=fill_value, inplace=True)
    
    return df_processed


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds year column as integer from title string.

    Parameters:
    -----------
    df : Movies pandas DataFrame
        The DataFrame to process

    Returns:
    --------
    pandas DataFrame
        The DataFrame with additional year column based on title.
    """
    df_return = df.copy()
    df_return["year"] = df_return.title.str[-6:].str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df_return = df_return[df_return["year"].str.isdigit()]
    df_return["year"] = df_return["year"].astype(int)
    return df_return


def transform_title_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms title column removing the year information.

    Parameters:
    -----------
    df : Movies pandas DataFrame
        The DataFrame to process

    Returns:
    --------
    pandas DataFrame
        The DataFrame with title column transformed.
    """
    df_return = df.copy()
    df_return["title"] = df_return["title"].str[:-6]
    return df_return


def transform_genres_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms genres column separating the genres by comma.

    Parameters:
    -----------
    df : Movies pandas DataFrame
        The DataFrame to process

    Returns:
    --------
    pandas DataFrame
        The DataFrame with genres column transformed.
    """
    df_return = df.copy()
    df_return["genres"] = df_return["genres"].str.replace("|", ", ", regex=False)
    return df_return


def pipeline_transform(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, df_tags: pd.DataFrame) -> pd.DataFrame:
    df_tags_agg = agg_user_movie_tags(df_tags)
    df_ratings_tags = df_ratings.merge(df_tags_agg, on=["userId", "movieId"], how="left")
    # Define column-specific rules
    column_rules = {
        'rating': 0,                 # Use median for ratings
        'user_movie_tags': 'No tags',              # Use specific value for tags
        'category': 'Unknown',              # Use specific value for category
        'views': lambda x: x.mean() * 0.8   # Use custom function (80% of mean)
    }
    df_movies_genre = add_year_column(df_movies)
    df_movies_transf_title = transform_title_column(df_movies_genre)
    df_movies_transformed = transform_genres_column(df_movies_transf_title)
    df_preprocess = df_ratings_tags.merge(df_movies_transformed, on="movieId", how='left')
    df_return = handle_missing_values(df_preprocess, column_rules=column_rules)
    df_return.columns = [col.lower() for col in df_return.columns]
    return df_return
