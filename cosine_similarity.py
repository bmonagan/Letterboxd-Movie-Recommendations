import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse




# Load data at startup
meta_data_file_location = "data/movie_metadata.parquet"
movie_vectors_data = "data/tfidf_matrix.npz"
metadata = pd.read_parquet(meta_data_file_location)
movie_vectors = sparse.load_npz(movie_vectors_data)

def get_recommendations(target_movie_title: str, num_recommendations: int = 10):
    # Ensure the target movie title exists in the metadata
    if target_movie_title not in metadata['title'].values:
        raise ValueError("Movie title not found.")
    
    # Find the index of the movie based on the title and the relative shared index
    row_index = metadata.index[metadata['title'] == target_movie_title][0]
    target_movie_vector = movie_vectors[row_index].reshape(1, -1)
    cosine_sim_scores = cosine_similarity(target_movie_vector, movie_vectors).flatten()
    similarity_series = pd.Series(cosine_sim_scores)
    sorted_similarities = similarity_series.sort_values(ascending=False)
    top_similar_movies_indices = sorted_similarities.index[1:num_recommendations+1]
    # Create and return a list of recommendations
    recommendations = []
    for idx in top_similar_movies_indices:
        rec_id = metadata.iloc[idx, 0]
        rec_title = metadata.iloc[idx, 1]
        rec_score = sorted_similarities.loc[idx]
        recommendations.append({
            "id": int(rec_id),
            "title": rec_title,
            "similarity": float(rec_score)
        })
    return recommendations