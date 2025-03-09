import sys
import os
import fire

from read import ReadData
from transform import EmbeddingGenerator
from training import EnhancedRecommendationPreComputer
from model_registry import register_recommendation_model

sys.path.append(os.path.abspath("..\\"))
from utils import get_logger, load_config

logger = get_logger(__name__)


def run_training_pipeline():
    """
    """

    # Load the config
    config = load_config("..\\config.yaml")

    # Read
    df = ReadData(
        feature_store_name=config["feature_store"]["name"], 
        feature_store_key=config["HOPSWORKS_API_KEY"], 
        feature_store_version=config["feature_store"]["version"]
    ).read_and_update_data()
    df.columns = ['user_id', 'item_id', 'rating', 'user_movie_tags', 'title', 'genres', 'year']
    items_df = ReadData.transform_items_df(df)
    user_interactions_df = ReadData.transform_user_df(df)

    # Transform
    embedding_generator = EmbeddingGenerator()
    # Generate item embeddings
    item_embeddings = embedding_generator.create_item_embeddings(items_df)
    # Generate user embeddings
    user_embeddings = embedding_generator.create_user_embeddings(user_interactions_df)

    # Train
    computer = EnhancedRecommendationPreComputer(
        items_df=items_df,
        user_profiles_df=user_interactions_df,
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        content_weight = float(config["model_hyperparameters"]["content_weight"]),
        collaborative_weight = float(config["model_hyperparameters"]["collaborative_weight"]),
        diversity_weight = float(config["model_hyperparameters"]["diversity_weight"]),
        base_score = float(config["model_hyperparameters"]["base_score"]),
        n_precomputed_recs = int(config["model_hyperparameters"]["n_precomputed_recs"]),
        similar_users = int(config["model_hyperparameters"]["similar_users"]),
    )

    # Compute all recommendations
    output_path=config["REC_OUTPUT_PATH"]
    computer.compute_all_recommendations(
        output_path=output_path,
        metadata_filename=config["METADATA_FILENAME"],
        recommendation_filename=config["RECOMMENDATIONS_FILENAME"]
        )

    # Register the model with MLFlow
    run_id = register_recommendation_model(
        recommender=computer,
        output_path=output_path,
        tracking_uri=config["MLFLOW_SERVER"],
        experiment_name=config["EXPERIMENT_NAME"],
        model_name=config["MODEL_NAME"],
        local_model_path=config["MODEL_LOCAL_STORAGE_PATH"],
        metadata_filename=config["METADATA_FILENAME"],
        recommendation_filename=config["RECOMMENDATIONS_FILENAME"]
    )
    
    logger.info(f"Model registered with MLFlow. Run ID: {run_id}")
    logger.info("You can now serve this model for inference using MLFlow.")

    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    fire.Fire(run_training_pipeline())
