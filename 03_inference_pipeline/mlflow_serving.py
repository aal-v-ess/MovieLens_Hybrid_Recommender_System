import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient


class MLFlowRecommendationServer:
    """Serves recommendations using MLFlow model with feedback handling and cold-start support"""
    
    def __init__(self, 
                 model_name: str = "hybrid_recommendation_model",
                 model_stage: str = "Production",
                 tracking_uri: Optional[str] = None,
                 feedback_store: Optional[str] = None):
        """
        Initialize server with MLFlow model and feedback storage
        
        Parameters:
        -----------
        model_name : str
            Name of the registered MLFlow model
        model_stage : str
            Stage of the model to use (Production, Staging, None, etc.)
        tracking_uri : str, optional
            URI for MLFlow tracking server
        feedback_store : str, optional
            Path to store user feedback
        """
        print(f"Loading MLFlow model {model_name} ({model_stage})...")
        
        # Set MLFlow tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Load the MLFlow model
        try:
            self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
            print(f"Successfully loaded MLFlow model: {model_name} ({model_stage})")
        except Exception as e:
            print(f"Error loading MLFlow model: {e}")
            print("Attempting to load latest version...")
            
            # Try loading the latest version
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = max([int(v.version) for v in versions])
                self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
                print(f"Successfully loaded model version: {latest_version}")
            else:
                raise ValueError(f"Could not find any versions of model {model_name}")
        
        # Initialize feedback storage
        self.feedback_store = feedback_store or 'recommendation_data/feedback.csv'
        self._init_feedback_store()
        
        # Pre-load cold start recommendations
        cold_start_result = self.model.predict(pd.DataFrame([{"user_id": "COLD_START", "n_recommendations": 100}]))
        self.cold_start_recs = pd.DataFrame(cold_start_result)
        
        print(f"Loaded recommendation model and {len(self.cold_start_recs)} cold start recommendations")
    
    def _init_feedback_store(self):
        """Initialize feedback storage"""
        if not Path(self.feedback_store).exists():
            pd.DataFrame(columns=[
                'user_id', 
                'item_id', 
                'rating', 
                'user_movie_tags',  # Store as dictionary string
                'timestamp', 
                'recommendation_type'
            ]).to_csv(self.feedback_store, index=False)
    
    def get_recommendations(self, 
                          user_id: str,
                          n_recommendations: int = 10,
                          randomize: bool = True,
                          user_data: Optional[Dict] = None) -> Dict:
        """Get recommendations with cold-start handling"""
        start_time = time.time()
        
        # Ensure user_id is string
        user_id = str(user_id)
        
        try:
            # Check if we should use cold start recommendations
            if user_id == "COLD_START" or user_data and user_data.get("is_new_user", False):
                recommendations = self._get_cold_start_recommendations(
                    n_recommendations, user_data
                )
                rec_type = 'cold_start'
            else:
                # Get personalized recommendations from the model
                model_input = pd.DataFrame([{
                    "user_id": user_id,
                    "n_recommendations": n_recommendations * 2  # Request more to allow for randomization
                }])
                
                # Get recommendations from the MLFlow model
                model_recommendations = self.model.predict(model_input)
                recommendations = self._process_recommendations(
                    model_recommendations, n_recommendations, randomize
                )
                rec_type = 'personalized'
            
            return {
                'recommendations': recommendations,
                'metadata': {
                    'serving_time': time.time() - start_time,
                    'user_id': user_id,
                    'recommendation_type': rec_type,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return {
                'error': str(e),
                'recommendations': [],
                'metadata': {
                    'serving_time': time.time() - start_time,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _process_recommendations(self, 
                               model_recommendations: List[Dict],
                               n_recommendations: int,
                               randomize: bool) -> List[Dict]:
        """Process recommendations from MLFlow model"""
        # Convert to DataFrame for easier manipulation
        recs_df = pd.DataFrame(model_recommendations)
        
        # Apply randomization if requested
        if randomize and 'score' in recs_df.columns:
            noise = np.random.normal(0, 0.1, size=len(recs_df))
            recs_df['randomized_score'] = recs_df['score'] + noise
            recs_df = recs_df.nlargest(n_recommendations, 'randomized_score')
        else:
            # Sort by rank or score
            if 'rank' in recs_df.columns:
                recs_df = recs_df.nsmallest(n_recommendations, 'rank')
            elif 'score' in recs_df.columns:
                recs_df = recs_df.nlargest(n_recommendations, 'score')
            else:
                recs_df = recs_df.head(n_recommendations)
        
        # Convert to records and ensure numeric types are Python native
        recommendations = recs_df.to_dict('records')
        for rec in recommendations:
            if 'score' in rec:
                rec['score'] = float(rec['score'])
            if 'rank' in rec:
                rec['rank'] = int(rec['rank'])
        
        return recommendations
    
    def _get_cold_start_recommendations(self,
                                      n_recommendations: int,
                                      user_data: Optional[Dict] = None) -> List[Dict]:
        """Get cold start recommendations, optionally using user data"""
        filtered_recs = self.cold_start_recs.copy()
        
        # Apply genre filtering if user data is provided
        if user_data and 'preferred_genres' in user_data and user_data['preferred_genres']:
            genre_filter = filtered_recs['genres'].apply(
                lambda x: any(genre in x for genre in user_data['preferred_genres'])
                if isinstance(x, str) else False
            )
            filtered_recs = filtered_recs[genre_filter]
            
            # If we have enough recommendations after filtering, use them
            if len(filtered_recs) >= n_recommendations:
                return filtered_recs.head(n_recommendations).to_dict('records')
        
        # Otherwise, return standard cold start recommendations
        return self.cold_start_recs.head(n_recommendations).to_dict('records')
    
    def record_feedback(self,
                       user_id: str,
                       item_id: int,
                       rating: float,
                       recommendation_type: Optional[str] = 'personalized',
                       user_movie_tags: Optional[List[str]] = None):
        """Record user feedback for future improvements"""
        feedback = pd.DataFrame([{
            'user_id': str(user_id),
            'item_id': item_id,
            'rating': rating,
            'user_movie_tags': json.dumps(user_movie_tags) if user_movie_tags else '[]',
            'timestamp': datetime.now().isoformat(),
            'recommendation_type': recommendation_type
        }])
        
        feedback.to_csv(self.feedback_store, mode='a', header=False, index=False)
        print(f"Recorded feedback for user {user_id} on item {item_id}")