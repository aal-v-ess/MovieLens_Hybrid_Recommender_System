from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
from movie_recommender_system.Notebooks.recommender_server import EnhancedRecommendationServer

# The current implementation does not include caching.
# In order to improve even more and be closer to real time recommendation this a next step.

class UserData(BaseModel):
    preferred_genres: Optional[List[str]] = Field(default=None, description="List of preferred genres")
    preferred_tags: Optional[List[str]] = Field(default=None, description="List of preferred tags")

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
    recommendations: List[Recommendation]
    metadata: Dict

app = FastAPI(
    title="Movie Recommender API",
    description="API for serving personalized movie recommendations",
    version="1.0.0"
)

# Global recommender instance
recommender = None

@app.on_event("startup")
async def startup_event():
    """Initialize recommender system on startup"""
    global recommender
    recommender = EnhancedRecommendationServer(
        recommendations_path='recommendation_data/recommendations.csv'
    )
    print("Recommender system initialized")

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
    preferred_tags: Optional[List[str]] = Query(None)
):
    """Get recommendations for a user"""
    try:
        user_data = None
        if preferred_genres or preferred_tags:
            user_data = UserData(
                preferred_genres=preferred_genres,
                preferred_tags=preferred_tags
            )
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            n_recommendations=n_recommendations,
            randomize=randomize,
            user_data=user_data.dict() if user_data else None
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_feedback(feedback: FeedbackData):
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
async def get_stats():
    """Get basic statistics about the recommender system"""
    return {
        "total_recommendations": len(recommender.recommendations_df),
        "total_users": len(recommender.user_recs.groups),
        "cold_start_recommendations": len(recommender.cold_start_recs),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)