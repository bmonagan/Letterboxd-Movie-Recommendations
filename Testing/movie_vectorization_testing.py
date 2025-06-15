from tmdbv3api import TMDb, Movie
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import os 
from dotenv import load_dotenv

load_dotenv()
tmdb = TMDb()
tmdb.api_key = os.environ.get('TMDB_API_KEY')
if tmdb.api_key is None:
    raise ValueError("TMDB_API_KEY is not set. Please set it as an environment variable.")


tmdb.language = 'en'

tmdb.region = 'US'


movie = Movie()
search_results = movie.search('Inception')

vectorizer = TfidfVectorizer()

if search_results:
    first_result = search_results[0]
    title = first_result.title
    genres = first_result.genre_ids
    overview = first_result.overview
    director = first_result.director if hasattr(first_result, 'director') else None
    actors = first_result.actors if hasattr(first_result, 'actors') else None

    # Combine the text data into a single string for vectorization
    text_data = f"{title} {' '.join(map(str, genres))} {overview} {director if director else ''} {' '.join(actors) if actors else ''}"

    # Fit the vectorizer on the text data
    tfidf_matrix = vectorizer.fit_transform([text_data])
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()

    # Convert the dense matrix to a DataFrame for better readability
    

    df = pd.DataFrame(dense, columns=feature_names)
    print(df)