from tmdbv3api import TMDb, Movie
import os 
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the TMDb object
tmdb = TMDb()

tmdb.api_key = os.environ.get('TMDB_API_KEY')

if tmdb.api_key is None:
    raise ValueError("TMDB_API_KEY is not set. Please set it as an environment variable.")


tmdb.language = 'en'

tmdb.region = 'US'

# Create a Movie object
movie = Movie()

# Search for movies matching the query 'Inception'
search_results = movie.search('Jurassic Park')
print(search_results)

# Print the title and release date of the first result
if search_results:
    first_result = search_results[0]
    print(f"Title: {first_result.title}")
    print(f"ID: {first_result.id}") # The ID is very important!
    print(f"Release Date: {first_result.release_date}")
    print(f"Overview: {first_result.overview}")
    print(f"Genres: {first_result.genre_ids}")  # List of genre IDs
else:
    print("No movie found with that name.")

# Expected Output:
# Title: Inception
# ID: 27205
# Release Date: 2010-07-15
# Overview: Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets...

#working so far. might not be the correct data source for this project.