import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from bs4 import BeautifulSoup
import requests

from helper_functions import clean_film_title, capitalize_roman



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

##TODO: BROKEN RIGHT NOW. INFINITE LOOP THROUGH LETTERBOXD USER'S WATCHED FILMS IF FILM NOT FOUND IN DATASET
def letter_boxd_get_recommendations(user_name: str, num_recommendations: int = 5, metadata=metadata, individual_recommendations: int = 5):

    # Fetch the user's watched films from Letterboxd
    url = f"https://letterboxd.com/{user_name}/films/diary/"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError("Failed to fetch data from Letterboxd. Please check the username or your internet connection.")
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract watched films
    lb_watched = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith(f'/{user_name}/film/'):
            film_id = href.split('/')[3].replace("-", " ")
            lb_watched.append(film_id)
    
    if not lb_watched:
        raise ValueError("No watched films found for this user.")
    
    # Get recommendations based on the watched films
    recommendations = []
    seen = set()
    # while lb_watched and len(recommendations) < num_recommendations or len(recommendations) + len(seen) == len(lb_watched):
    #     # Limit the number of films to process to avoid excessive API calls

    for film in lb_watched:
        # Clean the film title, removing trailing year if present and special characters 
        film = clean_film_title(film)
        film = film.title()
        film = capitalize_roman(film)

        if film in metadata['title'].values and film not in seen:
            seen.add(film)
            print(f"Processing film: {film}")
            # Get recommendations for each watched film
            recs = get_recommendations(film, num_recommendations)
            recommendations.append((film, recs))
        elif film not in metadata['title'].values and film not in seen:
            seen.add(film)
            print(f"Film '{film}' not found in the dataset. Skipping.")
        if len(recommendations) >= num_recommendations:
            break
    
    return recommendations