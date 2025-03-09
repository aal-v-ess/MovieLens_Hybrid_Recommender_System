import fire

from read import read_data
from transform import pipeline_transform
from validation import validate_features
from feature_store import to_feature_store
from utils import get_logger, load_config

logger = get_logger(__name__)


def run_features_pipeline():
    """
    Run the features pipeline
    """

    # Load the config
    config = load_config("..\\config.yaml")
    
    logger.info("Reading data...")

    # Read data
    df_ratings = read_data(path=config["DATA_PATH"], file_name=config["RATINGS_TABLE_NAME"], drop_columns=config["DROP_COLUMNS"])
    df_movies = read_data(path=config["DATA_PATH"], file_name=config["MOVIES_TABLE_NAME"])
    df_tags = read_data(path=config["DATA_PATH"], file_name=config["TAGS_TABLE_NAME"], drop_columns=config["DROP_COLUMNS"])

    logger.info("Transforming data...")

    # Transform data
    df_transform = pipeline_transform(df_ratings, df_movies, df_tags)

    # Write to feature store
    logger.info("Writing to feature store...")
    feature_group = to_feature_store(
        data=df_transform, 
        feature_store_api_key=config["HOPSWORKS_API_KEY"], 
        feature_store_name=config["feature_store"]["name"], 
        feature_group_version=config["feature_store"]["version"]
    )    
    logger.info(f"Feature pipeline complete. Created feature group version {feature_group.version}")



if __name__ == "__main__":
    fire.Fire(run_features_pipeline())
    