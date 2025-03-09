import hopsworks
import pandas as pd
from hsfs.feature_group import FeatureGroup


def to_feature_store(
    data: pd.DataFrame,
    feature_store_name: str = "movie_rec_sys",
    feature_store_api_key: str = None,
    feature_group_version: int = None,
) -> FeatureGroup:
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.
    """

    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=feature_store_api_key,
        project=feature_store_name,
    )
    fs = project.get_feature_store()

    # Check if feature group exists
    try:
        existing_fg = fs.get_feature_group(feature_store_name)
        # If we want a new version, increment the version number
        if feature_group_version is None:
            feature_group_version = existing_fg.version + 1
    except:
        # Feature group doesn't exist yet, start with version 1
        feature_group_version = 1 if feature_group_version is None else feature_group_version

    # Create feature group.
    movie_feature_group = fs.get_or_create_feature_group(
        name=feature_store_name,
        version=feature_group_version,
        description="MovieLens cleaned data for hybrid recommender system.",
        primary_key=["userid", "movieid"],
        online_enabled=False,
    )
    # Upload data.
    movie_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Update statistics.
    movie_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    movie_feature_group.update_statistics_config()
    movie_feature_group.compute_statistics()

    return movie_feature_group
