import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class EmbeddingGenerator:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.genre_vectorizer = TfidfVectorizer()
        self.tag_vectorizer = TfidfVectorizer()
        
    def _process_tag_dictionary(self, tag_dicts) -> np.ndarray:
        """Convert list of tag dictionaries to a frequency matrix."""
        # Get all unique tags
        all_tags = set()
        for tag_dict in tag_dicts:
            all_tags.update(tag_dict.keys())
        all_tags = sorted(list(all_tags))
        
        # Create matrix of tag frequencies
        tag_matrix = np.zeros((len(tag_dicts), len(all_tags)))
        for i, tag_dict in enumerate(tag_dicts):
            for j, tag in enumerate(all_tags):
                tag_matrix[i, j] = tag_dict.get(tag, 0)
        
        # Normalize the frequencies
        row_sums = tag_matrix.sum(axis=1, keepdims=True)
        tag_matrix = np.divide(tag_matrix, row_sums, where=row_sums!=0)
        
        return tag_matrix
    
    def create_item_embeddings(self, items_df) -> np.ndarray:
        """Generate embeddings for items using various features."""
        # 1. Generate text embeddings for titles
        title_embeddings = self.text_model.encode(items_df['title'].tolist())
        
        # 2. Process genres
        genre_embeddings = self.genre_vectorizer.fit_transform(items_df['genres']).toarray()
        
        # 3. Process year
        years = self.scaler.fit_transform(items_df[['year']]).reshape(-1)
        
        # 4. Process movie tags - now using the dictionary format
        tag_matrix = self._process_tag_dictionary(items_df['user_movie_tags'].tolist())
        
        # 5. Combine all embeddings
        combined_embeddings = np.hstack([
            title_embeddings,
            genre_embeddings,
            tag_matrix,
            years.reshape(-1, 1)
        ])
        
        return combined_embeddings
    
    def create_user_embeddings(self, user_profiles_df) -> np.ndarray:
        """Generate embeddings for users based on their aggregated profiles."""
        
        # 1. Process rating statistics
        rating_features = np.column_stack([
            user_profiles_df['rating_stats'].apply(lambda x: x['avg_rating']),
            user_profiles_df['rating_stats'].apply(lambda x: x['rating_std']),
            user_profiles_df['rating_stats'].apply(lambda x: x['num_ratings'])
        ])
        
        # 2. Process tag preferences
        # Create a vocabulary of all tags
        all_tags = set()
        for tags in user_profiles_df['tag_preferences']:
            all_tags.update(tags.keys())
        
        # Create tag frequency matrix
        tag_matrix = np.zeros((len(user_profiles_df), len(all_tags)))
        for i, tags in enumerate(user_profiles_df['tag_preferences']):
            for j, tag in enumerate(all_tags):
                tag_matrix[i, j] = tags.get(tag, 0)
        
        # Normalize tag frequencies
        tag_matrix = tag_matrix / (tag_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        # 3. Combine all features
        user_embeddings = np.hstack([
            rating_features,
            tag_matrix
        ])
        
        return user_embeddings
