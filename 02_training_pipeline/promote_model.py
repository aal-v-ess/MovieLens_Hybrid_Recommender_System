import mlflow
import argparse
from datetime import datetime

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

def main():
    """
    Command-line script to promote a model to production
    """
    parser = argparse.ArgumentParser(description="Promote an MLFlow model to production")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the registered model")
    parser.add_argument("--version", type=int, help="Specific version to promote (default: latest)")
    parser.add_argument("--tracking-uri", type=str, help="MLFlow tracking URI")
    
    args = parser.parse_args()
    
    promote_model_to_production(
        model_name=args.model_name,
        version=args.version,
        tracking_uri=args.tracking_uri
    )

if __name__ == "__main__":
    main()