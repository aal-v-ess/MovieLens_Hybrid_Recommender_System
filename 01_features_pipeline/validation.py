import pandas as pd
import great_expectations as ge


def validate_features(df: pd.DataFrame):
    """Validate features with Great Expectations"""

    df_validate = df.copy()
    df_validate["user_item_rating"] = df_validate["userid"].astype(str) + "_" + df_validate["movieid"].astype(str) + "_" + df_validate["rating"].astype(str)

    context = ge.get_context()
    ge_df = context.sources.pandas_default.read_dataframe(df_validate)

    # Define expectations
    expectations = [
        ge_df.expect_column_values_to_not_be_null("userid"),
        ge_df.expect_column_values_to_not_be_null("movieid"),
        ge_df.expect_column_values_to_not_be_null("rating"),
        ge_df.expect_column_values_to_not_be_null("year"),
        ge_df.expect_column_values_to_not_be_null("title"),
        ge_df.expect_column_values_to_not_be_null("genres"),
        ge_df.expect_column_values_to_not_be_null("title"),
        ge_df.expect_column_values_to_be_between("rating", min_value=0, max_value=5),
        ge_df.expect_column_values_to_be_between("year", min_value=1900, max_value=2100),
        ge_df.expect_column_values_to_not_be_null("title"),
        ge_df.expect_column_values_to_match_regex("genres", r"^[\w, ]*$", mostly=0.8),
        ge_df.expect_table_row_count_to_be_between(1, len(df_validate)),
        ge_df.expect_column_values_to_be_unique("user_item_rating")
    ]
    
    # Check if all validations passed
    passed = all(expectation.success for expectation in expectations)
    
    if not passed:
        failed = [exp for exp in expectations if not exp.success]
        print(f"Validation failed: {failed}")
    
    return passed
