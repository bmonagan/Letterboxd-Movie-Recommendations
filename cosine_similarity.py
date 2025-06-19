import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import json # Potentially needed if you were to load feature names from json

import helper_functions


# ID and Titles for Movie Name lookup
# Vectors for similarities
# They are on the same IDX making for an easy lookup.
meta_data_file_location = "data/movie_metadata.parquet"
movie_vectors = "data/tfidf_matrix.npz"

print(f"--- Loading data from '{movie_vectors}' and calculating cosine similarities ---")

# Load the Movie vectors into a df 
try:
    loaded_tfidf_df = sparse.load_npz(movie_vectors)
    print(f"'{movie_vectors}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{movie_vectors}' not found.")
    print("Please ensure you have run the data preprocessing and vectorization script to create this file.")

# Load the metadata into a DF 

try:
    metadata = pd.read_parquet(meta_data_file_location)
    print(f"'{metadata}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{metadata}' not found.")
    print("Please ensure you have run the data preprocessing and vectorization script to create this file.")
meta_data_first_col = metadata.columns[0]

target_movie_idx = helper_functions.movie_selection()

# Ensure the target index is within bounds of the loaded data
if target_movie_idx >= metadata['id'].max():
    print(f"Error: Target movie index {target_movie_idx} is out of bounds.")
    print("Please choose an index within the range [0, {}].".format(len(loaded_tfidf_df) - 1))
else:
    target_movie_id = metadata.iloc[target_movie_idx]
    target_movie_title = metadata.iloc[target_movie_idx]
    # Reshape the target movie's vector to (1, -1) for cosine_similarity function
    target_movie_vector = movie_vectors[target_movie_idx].reshape(1, -1)

    print(f"\nFinding recommendations for: ID={target_movie_id}, Title='{target_movie_title}'")

    # Calculate cosine similarities between the target movie vector and all other movie vectors
    # The result will be an array of similarity scores (1xN).
    cosine_sim_scores = cosine_similarity(target_movie_vector, movie_vectors)

    # Flatten the 1xN array to a 1D array of scores
    cosine_sim_scores = cosine_sim_scores.flatten()

    # Create a Pandas Series of similarity scores, using the DataFrame's original index
    similarity_series = pd.Series(cosine_sim_scores, index=loaded_tfidf_df.index)

    # Sort the movies by similarity score in descending order
    sorted_similarities = similarity_series.sort_values(ascending=False)

    # Get the top N similar movies (excluding the movie itself, which will have a similarity of 1.0)
    num_recommendations = 5
    # Skip the first element as it's the movie itself
    top_similar_movies_indices = sorted_similarities.index[1:num_recommendations+1]
    
    # movie_titles = pd.read_csv('data/TMDB_movie_dataset.csv')
    # selected_columns = ['id', 'title']
    # movie_titles =  movie_titles[selected_columns].copy()
    # movie_titles.set_index('id', inplace=True)  # <-- Add this line
    print(f"\nTop {num_recommendations} recommendations for '{target_movie_title}':")
    if top_similar_movies_indices.empty:
        print("No recommendations found (or only the movie itself was similar).")
    else:
        for idx in top_similar_movies_indices:
            rec_id = loaded_tfidf_df.loc[idx, 'id']
            rec_title = loaded_tfidf_df.loc[idx, 'title']
            rec_score = sorted_similarities.loc[idx]
            print(f"  - ID: {rec_id}, Title: '{rec_title}', Similarity Score: {rec_score:.4f}")

