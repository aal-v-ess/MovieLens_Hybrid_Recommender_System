from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
from datetime import datetime
import pandas as pd
from pathlib import Path
import mlflow
import subprocess
import sys

# Import the MLFlow-based recommendation server
from mlflow_serving import MLFlowRecommendationServer

class UserData(BaseModel):
    preferred_genres: Optional[List[str]] = Field(default=None, description="List of preferred genres")
    preferred_tags: Optional[List[str]] = Field(default=None, description="List of preferred tags")
    is_new_user: Optional[bool] = Field(default=False, description="Whether this is a new user")

class FeedbackData(BaseModel):
    user_id: str = Field(..., description="User ID")
    item_id: int = Field(..., description="Item ID")
    rating: float = Field(..., ge=0, le=5, description="Rating value between 0 and 5")
    user_movie_tags: Optional[List[str]] = Field(default=None, description="User tags for the movie")
    recommendation_type: Optional[str] = Field(default="personalized", description="Type of recommendation")

class Recommendation(BaseModel):
    item_id: int
    title: str
    genres: str
    score: float
    rank: int

class RecommendationResponse(BaseModel):
    recommendations: List[Dict]
    metadata: Dict

app = FastAPI(
    title="Movie Recommender API",
    description="API for serving personalized movie recommendations using MLFlow",
    version="1.1.0"
)

# Dependency for getting recommender
def get_recommender():
    return app.state.recommender

@app.on_event("startup")
async def startup_event():
    """Initialize recommender system on startup"""
    # MLFlow configuration
    mlflow_model_name = "hybrid_recommendation_model"
    mlflow_model_stage = "Production"
    mlflow_tracking_uri = "http://127.0.0.1:8080"
    model_uri = f"models:/{mlflow_model_name}/{mlflow_model_stage}"

    try:
        import cloudpickle
        if cloudpickle.__version__ != "3.1.1":
            print(f"Detected CloudPickle {cloudpickle.__version__}, but need 3.1.1. Attempting to install...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle==3.1.1", "--force-reinstall"])
            
            # Reload the module to get the new version
            if "cloudpickle" in sys.modules:
                del sys.modules["cloudpickle"]
            import cloudpickle
            print(f"CloudPickle version after install: {cloudpickle.__version__}")

        import pandas
        if pandas.__version__ < "2.0.0":
            print(f"Detected Pandas {pandas.__version__}, but need 2.1.4. Attempting to install...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudpickle==2.1.4", "--force-reinstall"])
            
            # Reload the module to get the new version
            if "pandas" in sys.modules:
                del sys.modules["pandas"]
            import pandas
            print(f"Pandas version after install: {pandas.__version__}")
    except Exception as e:
        print(f"Error handling CloudPickle and Pandas versions: {e}")
    
    # Initialize the MLFlow-based recommender
    app.state.recommender = MLFlowRecommendationServer(
        model_name=mlflow_model_name,
        model_stage=mlflow_model_stage,
        tracking_uri=mlflow_tracking_uri,
        feedback_store="recommendation_data/feedback.csv"
    )
    print("MLFlow-based recommender system initialized")

@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    n_recommendations: int = 10,
    randomize: bool = True,
    preferred_genres: Optional[List[str]] = Query(None),
    preferred_tags: Optional[List[str]] = Query(None),
    is_new_user: bool = False,
    recommender: MLFlowRecommendationServer = Depends(get_recommender)
):
    """Get recommendations for a user"""
    try:
        user_data = None
        if preferred_genres or preferred_tags or is_new_user:
            user_data = {
                "preferred_genres": preferred_genres,
                "preferred_tags": preferred_tags,
                "is_new_user": is_new_user
            }
        
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            randomize=randomize,
            user_data=user_data
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_feedback(
    feedback: FeedbackData,
    recommender: MLFlowRecommendationServer = Depends(get_recommender)
):
    """Record user feedback"""
    try:
        recommender.record_feedback(
            user_id=feedback.user_id,
            item_id=feedback.item_id,
            rating=feedback.rating,
            recommendation_type=feedback.recommendation_type,
            user_movie_tags=feedback.user_movie_tags
        )
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats(recommender: MLFlowRecommendationServer = Depends(get_recommender)):
    """Get basic statistics about the recommender system"""
    return {
        "mlflow_model_loaded": recommender.model is not None,
        "cold_start_recommendations_count": len(recommender.cold_start_recs),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)