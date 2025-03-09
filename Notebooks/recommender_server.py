import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Set
import json
from sklearn.metrics import r2_score


class EnhancedRecommendationServer:
    """Serves recommendations with feedback handling and cold-start support"""
    
    def __init__(self, 
                 recommendations_path: str = 'recommendation_data/recommendations.csv',
                 feedback_store: Optional[str] = None):
        """Initialize server with recommendations and feedback storage"""
        print("Loading recommendations...")
        self.recommendations_df = pd.read_csv(recommendations_path)
        
        # Convert user_id to string to ensure consistent type handling
        self.recommendations_df['user_id'] = self.recommendations_df['user_id'].astype(str)
        
        # Create indices for faster lookup
        self.user_recs = self.recommendations_df.groupby('user_id')
        
        # Initialize feedback storage
        self.feedback_store = feedback_store or 'recommendation_data/feedback.csv'
        self._init_feedback_store()
        
        # Load cold start recommendations
        self.cold_start_recs = self.recommendations_df[
            self.recommendations_df['user_id'] == 'COLD_START'
        ]
        
        print(f"Loaded {len(self.recommendations_df)} recommendations for {len(self.user_recs.groups)} users")
    
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
            # Check if user has existing recommendations
            if user_id in self.user_recs.groups:
                recommendations = self._get_personalized_recommendations(
                    user_id, n_recommendations, randomize
                )
                rec_type = 'personalized'
            else:
                # Use cold start recommendations
                recommendations = self._get_cold_start_recommendations(
                    n_recommendations, user_data
                )
                rec_type = 'cold_start'
            
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
    
    def _get_personalized_recommendations(self, 
                                        user_id: str,
                                        n_recommendations: int,
                                        randomize: bool) -> List[Dict]:
        """Get personalized recommendations for existing user"""
        user_recs = self.user_recs.get_group(user_id).copy()
        
        if randomize:
            noise = np.random.normal(0, 0.1, size=len(user_recs))
            user_recs['randomized_score'] = user_recs['score'] + noise
            user_recs = user_recs.nlargest(n_recommendations, 'randomized_score')
        else:
            user_recs = user_recs.nsmallest(n_recommendations, 'rank')
        
        # Convert to records and ensure numeric types are Python native
        recommendations = user_recs.to_dict('records')
        for rec in recommendations:
            rec['score'] = float(rec['score'])
            rec['rank'] = int(rec['rank'])
        
        return recommendations
    
    def _get_cold_start_recommendations(self,
                                      n_recommendations: int,
                                      user_data: Optional[Dict] = None) -> List[Dict]:
        """Get cold start recommendations, optionally using user data"""
        if user_data:
            filtered_recs = self.cold_start_recs.copy()
            
            # Filter by preferred genres if available
            if 'preferred_genres' in user_data:
                filtered_recs = filtered_recs[
                    filtered_recs['genres'].apply(
                        lambda x: any(genre in x for genre in user_data['preferred_genres'])
                    )
                ]
            
            # Filter by preferred tags if available
            if 'preferred_tags' in user_data and len(filtered_recs) >= n_recommendations:
                # Add tag similarity score
                filtered_recs['tag_score'] = filtered_recs['user_movie_tags'].apply(
                    lambda x: len(set(x).intersection(user_data['preferred_tags']))
                )
                filtered_recs = filtered_recs.nlargest(n_recommendations, 'tag_score')
            
            if len(filtered_recs) >= n_recommendations:
                return filtered_recs.head(n_recommendations).to_dict('records')
        
        # Fall back to general cold start recommendations
        return self.cold_start_recs.head(n_recommendations).to_dict('records')
    
    def record_feedback(self,
                       user_id: str,
                       item_id: int,
                       rating: float,
                       recommendation_type: Optional[str] = 'personalized',
                       user_movie_tags: Optional[str] = 'No tags'):
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