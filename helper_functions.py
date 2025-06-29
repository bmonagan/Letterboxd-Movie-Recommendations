import pandas as pd
import json
import ast
from tmdbv3api import TMDb, Movie
import os 
from dotenv import load_dotenv
import re

def parse_and_clean(text):
    if pd.isna(text) or text == '[]': 
        return ''
    try:
        
        list_of_dicts = json.loads(text)
        return ' '.join([d['name'] for d in list_of_dicts if 'name' in d])
    except (json.JSONDecodeError, TypeError):
        return str(text).replace('[', '').replace(']', '').replace("'", "").replace('"', '').replace(',', ' ').strip()



# Load environment variables from .env file
load_dotenv()

# Initialize the TMDb object
tmdb = TMDb()

tmdb.api_key = os.environ.get('TMDB_API_KEY')

if tmdb.api_key is None:
    raise ValueError("TMDB_API_KEY is not set. Please set it as an environment variable.")


tmdb.language = 'en'

tmdb.region = 'US'
def movie_selection():
    # Create a Movie object
    movie = Movie()
    try:
        user_movie = input("Please select a movie:")
        
    except:
        print("Invalid movie title")

    search_results = movie.search(user_movie)
    print(search_results)

    
    if search_results:
        first_result = search_results[0]
        print(f"Here is the ID of the Movie you selected: {first_result.id}")
        return first_result.id
    else:
        print("No movie found with that name.")



def clean_film_title(slug: str) -> str:
    """
    Remove a trailing "-YYYY" if present
    """
    return re.sub(r'\d{4}$', '', slug).replace("-", " ")

def capitalize_roman(title: str) -> str:
    """
    Capitalizes a Roman numeral at the end of a film title if present.
    Example: "Rocky ii" -> "Rocky II"
    """
    import re
    # Match a space and roman numeral at the end (case-insensitive)
    match = re.search(r'(.*\s)([ivx]+)$', title, re.IGNORECASE)
    if match:
        prefix = match.group(1)
        roman = match.group(2).upper()
        return f"{prefix}{roman}"
    return title
