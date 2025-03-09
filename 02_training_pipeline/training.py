import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional
from typing import List, Dict

sys.path.append(os.path.abspath("..\\"))
from utils import get_logger, load_config

logger = get_logger(__name__)


class EnhancedRecommendationPreComputer:
    def __init__(self, 
                 items_df: pd.DataFrame,
                 user_profiles_df: pd.DataFrame,
                 user_embeddings: np.ndarray,
                 item_embeddings: np.ndarray,
                 content_weight: Optional[float] = 0.3,
                 collaborative_weight: Optional[float] = 0.7,
                 diversity_weight: Optional[float] = 0.05,
                 base_score: Optional[float] = 0.05,
                 n_precomputed_recs: Optional[float] = 100,
                 similar_users: Optional[float] = 20,
                 cold_start_popularity: Optional[float] = 0.7,
                 cold_start_diversity: Optional[float] = 0.3,
                 batch_size: Optional[int]=100,
                 user_feedback_df: Optional[pd.DataFrame] = None,):
        """Initialize recommendation computer"""
        self.items_df = items_df
        self.user_profiles_df = user_profiles_df
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.user_feedback_df = user_feedback_df
        
        # Configuration
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.diversity_weight = diversity_weight
        self.base_score = base_score
        self.n_precomputed_recs = n_precomputed_recs
        self.similar_users = similar_users
        self.cold_start_popularity = cold_start_popularity
        self.cold_start_diversity = cold_start_diversity
        self.batch_size = batch_size

        # Calculate dimensions
        self.user_tag_start = 3  # After rating stats (mean, std, count)
        self.item_tag_start = 384 + 20  # After title embedding (384) and approximate genre embedding (20)
        
        # Initialize computations
        self._compute_genre_representation()
        self._compute_item_metrics()
        
        logger.info(f"User embedding shape: {self.user_embeddings.shape}")
        logger.info(f"Item embedding shape: {self.item_embeddings.shape}")
    
    def compute_user_recommendations(self, user_idx: int) -> List[Dict]:
        """Compute recommendations for a single user with bias mitigation"""
        # Get user profile
        user_profile = self.user_profiles_df.iloc[user_idx]
        
        # 1. Get user components
        user_tag_vector = self.user_embeddings[user_idx, self.user_tag_start:]
        
        # 2. Get item components
        item_tag_vectors = self.item_embeddings[:, self.item_tag_start:-1]  # Exclude year
        
        # Calculate tag-based similarity only on tag portions
        # Reshape both to 2D arrays for cosine_similarity
        user_tag_vector_2d = user_tag_vector.reshape(1, -1)
        min_dim = min(user_tag_vector_2d.shape[1], item_tag_vectors.shape[1])
        
        # Take only the common dimensions
        tag_similarity = cosine_similarity(
            user_tag_vector_2d[:, :min_dim],
            item_tag_vectors[:, :min_dim]
        )[0]
        
        # 3. Rating-based weighting
        rating_stats = user_profile['rating_stats']
        rating_weight = np.clip(rating_stats['avg_rating'] / 5.0, 0.5, 1.0)
        
        # 4. Genre diversity
        diversity_scores = self._calculate_diversity_scores()
        
        # 5. Collaborative filtering scores
        cf_scores = self._compute_collaborative_scores(user_idx)
        
        # Combine scores with weights
        final_scores = (
            self.content_weight * self._normalize_scores(tag_similarity) * rating_weight +
            self.collaborative_weight * self._normalize_scores(cf_scores) +
            self.diversity_weight * diversity_scores +
            self.base_score  # Base score
        )
        
        # Get top items
        top_indices = np.argsort(final_scores)[::-1][:self.n_precomputed_recs]
        
        # Create recommendations
        recommendations = []
        for rank, idx in enumerate(top_indices):
            recommendations.append({
                'user_id': user_idx,
                'item_id': self.items_df.index[idx],
                'title': self.items_df.iloc[idx]['title'],
                'genres': self.items_df.iloc[idx]['genres'],
                'score': float(final_scores[idx]),
                'rank': rank,
                'recommendation_type': 'personalized'
            })
        
        return recommendations
    
    def _compute_collaborative_scores(self, user_idx: int) -> np.ndarray:
        """Compute collaborative filtering scores for a user"""
        # Use only tag portions for user similarity
        user_tag_vector = self.user_embeddings[user_idx, self.user_tag_start:]
        all_user_tag_vectors = self.user_embeddings[:, self.user_tag_start:]
        
        # Calculate similarity using common dimensions
        min_dim = min(user_tag_vector.shape[0], all_user_tag_vectors.shape[1])
        user_similarities = cosine_similarity(
            user_tag_vector[:min_dim].reshape(1, -1),
            all_user_tag_vectors[:, :min_dim]
        )[0]
        
        # Get top similar users (excluding self)
        n_similar = self.similar_users
        similar_users = np.argsort(user_similarities)[-n_similar-1:-1][::-1]
        
        # Calculate weighted sum of similar users' interactions
        cf_scores = np.zeros(len(self.items_df))
        
        for sim_user in similar_users:
            sim_score = user_similarities[sim_user]
            sim_user_profile = self.user_profiles_df.iloc[sim_user]
            
            # Use rated movies
            rated_movies = sim_user_profile['rated_movies']
            for movie_id in rated_movies:
                if movie_id in self.items_df.index:
                    movie_idx = self.items_df.index.get_loc(movie_id)
                    cf_scores[movie_idx] += sim_score
        
        return cf_scores

    def _compute_genre_representation(self):
        """Compute genre representation metrics"""
        # Compute genre counts
        genre_counts = {}
        for _, row in self.items_df.iterrows():
            if isinstance(row['genres'], str):  # Ensure genres is a string
                genres = row['genres'].split(',')
                for genre in genres:
                    genre = genre.strip()  # Remove any whitespace
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Calculate genre representation scores
        total_items = len(self.items_df)
        self.genre_representation = {
            genre: count/total_items 
            for genre, count in genre_counts.items()
        }
        
        # Identify underrepresented genres
        mean_representation = np.mean(list(self.genre_representation.values()))
        self.underrepresented_genres = {
            genre: score 
            for genre, score in self.genre_representation.items()
            if score < mean_representation
        }
    
    def compute_cold_start_recommendations(self) -> pd.DataFrame:
        """
        Compute generic recommendations for new users based on popularity and diversity
        Returns DataFrame with cold start recommendations
        """
        logger.info("Computing cold-start recommendations...")
        
        # Create a scoring DataFrame for all items
        item_scores = pd.DataFrame(index=self.items_df.index)
        
        # Calculate popularity scores (if we have feedback data)
        if hasattr(self, 'item_popularity'):
            popularity_scores = self._normalize_series(self.item_popularity['mean'])
        else:
            # If no feedback data, use equal popularity
            popularity_scores = pd.Series(1.0/len(self.items_df), index=self.items_df.index)
        
        # Calculate diversity scores
        diversity_scores = self._calculate_diversity_scores()
        
        # Combine scores
        item_scores['popularity'] = popularity_scores
        item_scores['diversity'] = diversity_scores
        item_scores['final_score'] = (
            self.cold_start_popularity * item_scores['popularity'] +
            self.cold_start_diversity * item_scores['diversity']
        )
        
        # Get top recommendations
        top_items = item_scores.nlargest(self.n_precomputed_recs, 'final_score')
        
        # Create recommendations records
        cold_start_recs = []
        for rank, (idx, row) in enumerate(top_items.iterrows()):
            cold_start_recs.append({
                'user_id': 'COLD_START',
                'item_id': idx,
                'title': self.items_df.loc[idx, 'title'],
                'genres': self.items_df.loc[idx, 'genres'],
                'score': row['final_score'],
                'rank': rank,
                'recommendation_type': 'cold_start'
            })
        
        return pd.DataFrame(cold_start_recs)
    
    def _compute_item_metrics(self):
        """Compute various item metrics for cold-start and bias mitigation"""
        self.item_popularity = pd.DataFrame(index=self.items_df.index)
        self.item_popularity['count'] = 1
        self.item_popularity['mean'] = 0
    
    def _calculate_diversity_scores(self) -> pd.Series:
        """Calculate diversity scores for items based on genres"""
        diversity_scores = []
        
        for _, item in self.items_df.iterrows():
            if isinstance(item['genres'], str):  # Check if genres is a string
                genres = item['genres'].split(',')
                genres = [g.strip() for g in genres]  # Clean any whitespace
                
                # Higher score for underrepresented genres
                genre_scores = []
                for genre in genres:
                    if genre in self.genre_representation:
                        genre_scores.append(1/self.genre_representation[genre])
                
                diversity_scores.append(np.mean(genre_scores) if genre_scores else 0)
            else:
                diversity_scores.append(0)  # Default score for items without genres
        
        return pd.Series(diversity_scores, index=self.items_df.index)
    
    
    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Normalize series to [0,1] range"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val)

    
    def _normalize_scores(self, scores):
        """Normalize scores to range [0,1]"""
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def compute_all_recommendations(self, output_path: str = 'recommendation_data', metadata_filename: str = 'metadata', recommendation_filename: str = "recommendations"):
        """Compute recommendations for all users plus cold-start"""
        start_time = time.time()
        Path(output_path).mkdir(exist_ok=True)
        
        logger.info("Starting recommendation computation...")
        
        # First, compute cold-start recommendations
        cold_start_df = self.compute_cold_start_recommendations()
        cold_start_df.to_csv(f'{output_path}/{recommendation_filename}.csv', index=False)
        logger.info(f"Saved {len(cold_start_df)} cold-start recommendations")
        
        # Then compute personalized recommendations in batches
        batch_size = self.batch_size
        n_users = len(self.user_embeddings)
        
        logger.info(f"Computing personalized recommendations for {n_users} users...")
        for batch_start in range(0, n_users, batch_size):
            batch_end = min(batch_start + batch_size, n_users)
            logger.info(f"Processing users {batch_start} to {batch_end}...")
            
            batch_recommendations = []
            for user_idx in range(batch_start, batch_end):
                user_recs = self.compute_user_recommendations(user_idx)
                batch_recommendations.extend(user_recs)
            
            # Append batch to CSV
            batch_df = pd.DataFrame(batch_recommendations)
            batch_df.to_csv(
                f'{output_path}/{recommendation_filename}.csv', 
                mode='a', 
                header=False, 
                index=False
            )
        
        computation_time = time.time() - start_time
        logger.info(f"Completed recommendation computation in {computation_time:.2f} seconds")
        
        # Save metadata
        metadata = {
            'computation_time': computation_time,
            'n_users': n_users,
            'n_items_per_user': self.n_precomputed_recs,
            'timestamp': datetime.now().isoformat(),
            'cold_start_items': len(cold_start_df),
            'genre_representation': self.genre_representation
        }
        
        pd.DataFrame([metadata]).to_json(f'{output_path}/{metadata_filename}.json')
        logger.info("Saved computation metadata")