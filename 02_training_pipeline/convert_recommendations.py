import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse

def convert_existing_recommendations(input_path, output_path):
    """
    Convert existing recommendation data to be compatible with MLFlow model
    
    Parameters:
    -----------
    input_path : str
        Path to the existing recommendations CSV
    output_path : str
        Path to save the converted recommendations
    """
    print(f"Converting recommendations from {input_path} to {output_path}")
    
    # Load existing recommendations
    recommendations_df = pd.read_csv(input_path)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as-is for now (format is already compatible)
    recommendations_df.to_csv(output_path, index=False)
    
    print(f"Converted {len(recommendations_df)} recommendations")
    print(f"Unique users: {recommendations_df['user_id'].nunique()}")
    
    # Check for cold start recommendations
    cold_start_count = len(recommendations_df[recommendations_df['user_id'] == 'COLD_START'])
    print(f"Cold start recommendations: {cold_start_count}")
    
    return recommendations_df

def main():
    parser = argparse.ArgumentParser(description="Convert existing recommendation data for MLFlow")
    parser.add_argument("--input", type=str, default="recommendation_data/recommendations.csv",
                        help="Path to existing recommendations CSV")
    parser.add_argument("--output", type=str, default="mlflow_data/recommendations.csv",
                        help="Path to save converted recommendations")
    
    args = parser.parse_args