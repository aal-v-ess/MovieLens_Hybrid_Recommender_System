import os
import sys
import mlflow
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.append(os.path.abspath("..\\"))
from utils import get_logger, load_config

logger = get_logger(__name__)


class RecommendationModelRegistry:
    """
    A wrapper class to handle MLFlow registration for recommendation models
    """
    def __init__(self, 
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "MovieLens_Hybrid_Recommender_System",
                 model_name: str = "hybrid_recommendation_model"):
        """
        Initialize the MLFlow registry handler
        
        Parameters:
        -----------
        tracking_uri : str, optional
            URI for MLFlow tracking server. If None, local file storage will be used.
        experiment_name : str
            Name of the MLFlow experiment to use
        model_name : str
            Name to register the model under in the MLFlow registry
        """
        # Set MLFlow tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get the experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        self.model_name = model_name
    
    def log_and_register_model(self, 
                              recommender_model: Any,
                              metrics: Dict[str, float],
                              params: Dict[str, Any],
                              local_model_path: str,
                              artifacts: Dict[str, str] = None,
                              register: bool = True,
                              tags: Dict[str, str] = None):
        """
        Log and optionally register a recommendation model in MLFlow
        
        Parameters:
        -----------
        recommender_model : Any
            The recommendation model instance to register
        metrics : Dict[str, float]
            Performance metrics to log (e.g., computation time)
        params : Dict[str, Any]
            Model parameters to log
        artifacts : Dict[str, str], optional
            Additional artifacts to log (path mapping)
        register : bool
            Whether to register the model in the model registry
        tags : Dict[str, str], optional
            Tags to attach to the run
        
        Returns:
        --------
        str
            Run ID of the logged model
        """
        # Start a new MLFlow run
        with mlflow.start_run(run_name='movielens_rec_sys') as run:
            run_id = run.info.run_id
            
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log any additional tags
            if tags:
                mlflow.set_tags(tags)
            
            # Save the model
            temp_model_path = local_model_path
            Path(temp_model_path).mkdir(exist_ok=True)
            
            # Save the recommender model
            with open(f"{temp_model_path}/{self.model_name}.pkl", "wb") as f:
                pickle.dump(recommender_model, f)
            
            # Save the model class definition as a Python file
            model_code = recommender_model.__class__.__module__
            if hasattr(recommender_model.__class__, "__file__"):
                model_code_path = recommender_model.__class__.__file__
                if os.path.exists(model_code_path):
                    mlflow.log_artifact(model_code_path, "code")
            
            # Log custom artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, name)
            
            # Log the pickled model
            mlflow.log_artifact(f"{temp_model_path}/{self.model_name}.pkl", "model")
            
            # Define a proper wrapper class for MLFlow
            class RecommendationModelWrapper(mlflow.pyfunc.PythonModel):
                def __init__(self, model):
                    self.model = model
                
                def load_context(self, context):
                    # Nothing to do here as we already have the model
                    pass
                
                def predict(self, context, model_input):
                    """
                    Wrapper function for MLFlow model serving
                    
                    Inputs:
                    -------
                    model_input : pandas.DataFrame
                        DataFrame with user_id and optionally n_recommendations
                    
                    Returns:
                    --------
                    list
                        Recommendations for the user
                    """
                    # Convert to dict if DataFrame
                    if isinstance(model_input, pd.DataFrame):
                        if len(model_input) == 0:
                            return []
                        
                        # Get first row as dict
                        model_input = model_input.iloc[0].to_dict()
                    
                    user_id = model_input.get("user_id")
                    n_recommendations = model_input.get("n_recommendations", 10)
                    
                    # Check for cold start
                    if user_id == "COLD_START":
                        cold_start_recs = self.model.compute_cold_start_recommendations()
                        return cold_start_recs.head(n_recommendations).to_dict(orient="records")
                    
                    # Convert to index if necessary
                    if isinstance(user_id, str) and user_id.isdigit():
                        user_id = int(user_id)
                    
                    # Get user index
                    user_idx = user_id
                    if hasattr(self.model, 'user_profiles_df'):
                        if user_id in self.model.user_profiles_df.index:
                            user_idx = self.model.user_profiles_df.index.get_loc(user_id)
                    
                    # Get recommendations
                    user_recs = self.model.compute_user_recommendations(user_idx)
                    return user_recs[:n_recommendations]
            
            # Create an example input
            example_input = pd.DataFrame([{
                "user_id": 0,
                "n_recommendations": 5
            }])
            
            # Create the wrapper instance
            model_wrapper = RecommendationModelWrapper(recommender_model)
            
            # Create requirements list
            pip_requirements = [
                "numpy>=1.20.0",
                "pandas==2.1.4",
                "scikit-learn>=0.24.2",
                "mlflow>=2.0.0",
                "cloudpickle==3.1.1"
            ]
            
            # Log the model using the Python function flavor
            mlflow.pyfunc.log_model(
                artifact_path="hybrid_recommendation_model",
                python_model=model_wrapper,
                artifacts={"recommender": f"{temp_model_path}/{self.model_name}.pkl"},
                pip_requirements=pip_requirements,
                input_example=example_input
            )
            
            # Register the model if requested
            if register:
                model_uri = f"runs:/{run_id}/hybrid_recommendation_model"
                version = mlflow.register_model(model_uri, self.model_name)
                logger.info(f"Registered model: {self.model_name}, version: {version.version}")
        
        return run_id
    




def promote_model_to_production(model_name, version=None, tracking_uri=None):
    """
    Promote a specific model version to production
    
    Parameters:
    -----------
    model_name : str
        Name of the registered model
    version : str or int, optional
        Specific version to promote. If None, promotes the latest version
    tracking_uri : str, optional
        MLFlow tracking URI
    
    Returns:
    --------
    dict
        Information about the promoted model
    """
    # Set tracking URI if provided
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Get MLFlow client
    client = mlflow.tracking.MlflowClient()
    
    # Find the model versions
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    
    # If version not specified, get the latest version
    if version is None:
        version = max([int(mv.version) for mv in model_versions])
    
    # Find the specified version
    model_version = None
    for mv in model_versions:
        if mv.version == str(version):
            model_version = mv
            break
    
    if model_version is None:
        raise ValueError(f"Version {version} not found for model '{model_name}'")
    
    # Archive any existing production models
    for mv in model_versions:
        if mv.current_stage == "Production" and mv.version != str(version):
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived",
                archive_existing_versions=False
            )
            print(f"Archived previous production model: {model_name} version {mv.version}")
    
    # Promote the specified version to production
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production"
    )
    
    # Add description and update time
    client.update_model_version(
        name=model_name,
        version=str(version),
        description=f"Promoted to production on {datetime.now().isoformat()}"
    )
    
    print(f"Successfully promoted {model_name} version {version} to Production")
    
    # Return the promoted model version info
    return {
        "name": model_name,
        "version": version,
        "stage": "Production",
        "promotion_time": datetime.now().isoformat()
    }






def register_recommendation_model(recommender, 
                                 output_path: str = "recommendation_data",
                                 tracking_uri: Optional[str] = None,
                                 experiment_name: str = "recommendation_system",
                                 model_name: str = "hybrid_recommendation_model",
                                 local_model_path: str = "model_storage",
                                 metadata_filename: str = "metadata",
                                 recommendation_filename: str = "recommendations"):
    """
    Helper function to register an EnhancedRecommendationPreComputer model with MLFlow
    
    Parameters:
    -----------
    recommender : EnhancedRecommendationPreComputer
        The recommendation model to register
    output_path : str
        Path where recommendation outputs are stored
    tracking_uri : str, optional
        MLFlow tracking URI
    experiment_name : str
        MLFlow experiment name
    model_name : str
        Model name in registry
    
    Returns:
    --------
    str
        Run ID of the registered model
    """
    # Create model registry handler
    registry = RecommendationModelRegistry(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        model_name=model_name
    )
    
    # Get model parameters
    params = {
        "content_weight": recommender.content_weight,
        "collaborative_weight": recommender.collaborative_weight,
        "diversity_weight": recommender.diversity_weight,
        "base_score": recommender.base_score,
        "n_precomputed_recs": recommender.n_precomputed_recs,
        "similar_users": recommender.similar_users,
        "cold_start_popularity": recommender.cold_start_popularity,
        "cold_start_diversity": recommender.cold_start_diversity,
        "batch_size": recommender.batch_size,
        "user_count": len(recommender.user_profiles_df),
        "item_count": len(recommender.items_df),
        "timestamp": datetime.now().isoformat()
    }
    
    # Look for metrics in the output path
    metrics = {}
    metadata_path = Path(output_path) / f"{metadata_filename}.json"
    if metadata_path.exists():
        try:
            metadata = pd.read_json(metadata_path).iloc[0].to_dict()
            metrics = {
                "computation_time": metadata.get("computation_time", 0),
                "n_users": metadata.get("n_users", 0),
                "n_items_per_user": metadata.get("n_items_per_user", 0),
                "cold_start_items": metadata.get("cold_start_items", 0)
            }
        except Exception as e:
            logger.info(f"Error reading metadata: {e}")
    
    # Collect artifacts
    artifacts = {}
    recommendations_path = Path(output_path) / f"{recommendation_filename}.csv"
    if recommendations_path.exists():
        artifacts["recommendations"] = str(recommendations_path)
    
    # Add tags
    tags = {
        "model_type": "recommendation_system",
        "framework": "custom",
        "embedding_dimensions": str(recommender.user_embeddings.shape[1]),
        "timestamp": datetime.now().isoformat()
    }
    
    # Log and register the model
    run_id = registry.log_and_register_model(
        recommender_model=recommender,
        metrics=metrics,
        params=params,
        artifacts=artifacts,
        register=True,
        tags=tags,
        local_model_path=local_model_path
    )

    model_info = dict()
    model_info = promote_model_to_production(model_name=model_name, version=None, tracking_uri=tracking_uri)
    logger.info(f"Model registered and promoted to production: {model_info}")


    
    return run_id