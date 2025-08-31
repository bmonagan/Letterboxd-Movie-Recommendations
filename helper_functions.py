# helper_functions.py
"Helper functions for movie data processing and TMDb interaction."
# Standard library imports
import re
import os
import json
import pandas as pd

from tmdbv3api import TMDb, Movie

from dotenv import load_dotenv


def parse_and_clean(text):
    "Parse a string representation of a list of dictionaries and return a cleaned string of names."
    if pd.isna(text) or text == '[]':
        return ''
    try:
        list_of_dicts = json.loads(text)
        return ' '.join([d['name'] for d in list_of_dicts if 'name' in d])
    except (json.JSONDecodeError, TypeError):
        chars_to_remove = ['[', ']', '{', '}', '"', "'"]
        for char in chars_to_remove:
            text = str(text).replace(char, '')
        return str(text).strip()


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
    "Interactively select a movie and return its TMDb ID."
    # Create a Movie object
    movie = Movie()
    try:
        user_movie = input("Please select a movie:")
    except EOFError:
        print("Input ended unexpectedly.")
        return None
    except KeyboardInterrupt:
        print("Input cancelled by user.")
        return None

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
    # Match a space and roman numeral at the end (case-insensitive)
    match = re.search(r'(.*\s)([ivx]+)$', title, re.IGNORECASE)
    if match:
        prefix = match.group(1)
        roman = match.group(2).upper()
        return f"{prefix}{roman}"
    return title
