import pandas as pd
import hopsworks
from pathlib import Path
from typing import Optional
from collections import Counter


class ReadData:
    def __init__(self, feature_store_name: str, feature_store_key: str, feature_store_version: int, feedback_store: Optional[str] = None):
        # Initialize feedback storage
        self.feedback_store = feedback_store or 'recommendation_data/feedback.csv'
        self.feature_store_name = feature_store_name or 'clean_movies_data.csv'
        self.feature_store_key = feature_store_key
        self.feature_store_version = feature_store_version
    #     self._init_feedback_store()

    # def _init_feedback_store(self):
    #     """Initialize feedback storage"""
    #     if not Path(self.feedback_store).exists():
    #         pd.DataFrame(columns=[
    #             'user_id', 'item_id', 'rating', 'user_movie_tags', 'timestamp', 'recommendation_type'
    #         ]).to_csv(self.feedback_store, index=False)

    def _read_movies_data(self):
        """Read movies data"""
        # df_return = pd.read_csv(self.feature_store_name)
        print("Reading movie feature store.")
        project = hopsworks.login(
            api_key_value=self.feature_store_key, project=self.feature_store_name
        )
        fs = project.get_feature_store(name='movie_rec_sys_featurestore')
        fg = fs.get_feature_group(self.feature_store_name, version=self.feature_store_version)
        df_return = fg.read()
        print("Read data from feature store columns:", df_return.columns)
        df_return.columns = ['user_id', 'item_id', 'rating', 'user_movie_tags', 'title', 'genres', 'year']
        return df_return
    
    def _read_feedback_data(self):
        return pd.read_csv(self.feedback_store)
    
    def _cat_ids(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
        """Categorize ids."""
        df[id] = pd.Categorical(df[id])
        df[id] = df[id].cat.codes
        return df
    
    def read_and_update_data(self):
        print("No feedback feature store. Reading movie data from feature store.")
        df_return = self._read_movies_data()
        df_return = self._cat_ids(df_return, 'user_id')
        df_return = self._cat_ids(df_return, 'item_id')
        return df_return
        # if len(self._read_feedback_data()) == 0:
        #     print("No feedback feature store. Reading movie data from feature store.")
        #     df_return = self._read_movies_data()
        #     df_return = self._cat_ids(df_return, 'user_id')
        #     df_return = self._cat_ids(df_return, 'item_id')
        #     return df_return
        # else:
        #     print("Feedback feature store available. Reading movie data and feedback from feature store.")
        #     df_movies = self._read_movies_data()
        #     df_movies = self._cat_ids(df_movies, 'user_id')
        #     df_movies = self._cat_ids(df_movies, 'item_id')
        #     df_feedback = self._read_feedback_data()
        #     df_feedback_enriched = pd.merge(df_feedback.drop(columns=['timestamp', 'recommendation_type']), 
        #                                     df_movies[['item_id', 'title', 'genres', 'year']], 
        #                                     on='item_id', 
        #                                     how='left').drop_duplicates()
        #     df_return = pd.concat([df_movies, df_feedback_enriched], ignore_index=True)
        #     return df_return
        
    def transform_items_df(df):
        df_items = df[['item_id', 'title', 'genres', 'year', 'user_movie_tags']].copy()
        tag_counts = df_items.groupby('item_id')['user_movie_tags'].agg(lambda x: dict(Counter(x))).reset_index()
        unique_movies = df_items.drop_duplicates(subset=['item_id']).drop(columns=['user_movie_tags'])
        df_return = pd.merge(unique_movies, tag_counts, on='item_id')
        return df_return
    
    def transform_user_df(df):
        """Transform user interactions into aggregated user profiles."""
        df_user = df[['user_id', 'item_id', 'rating', 'user_movie_tags']].copy()
        user_profiles = {}
        for user_id in df_user['user_id'].unique():
            user_data = df_user[df_user['user_id'] == user_id]
            
            # Aggregate ratings
            rating_stats = {
                'avg_rating': user_data['rating'].mean(),
                'rating_std': user_data['rating'].std(),
                'num_ratings': len(user_data),
                'rating_distribution': dict(Counter(user_data['rating']))
            }
            
            # Aggregate movie tags
            tag_counts = dict(Counter(user_data['user_movie_tags'].explode()))
            
            # Create user profile
            user_profiles[user_id] = {
                'user_id': user_id,
                'rating_stats': rating_stats,
                'tag_preferences': tag_counts,
                'rated_movies': user_data['item_id'].tolist()
            }
        
        return pd.DataFrame.from_dict(user_profiles, orient='index')
