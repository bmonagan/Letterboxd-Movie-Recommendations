# Letterboxd Movie Recommendations

A Python project and API for generating movie recommendations based on content similarity, using TMDB metadata and Letterboxd user data.

---

## Features

- **Content-Based Recommendations:**  
  Uses TF-IDF vectorization and cosine similarity to recommend movies similar to a given title.

- **Letterboxd User Integration:**  
  Fetches a user's recently watched films from Letterboxd and generates personalized recommendations.

- **FastAPI Web API:**  
  Provides endpoints for movie and user-based recommendations.

---

## How It Works

1. **Data Preparation:**  
   - TMDB movie metadata is loaded and cleaned.
   - Movie overviews, genres, and keywords are combined and vectorized using TF-IDF.
   - Vectors and metadata are split and saved for fast lookup.

2. **Recommendation Engine:**  
   - For a given movie title, the system finds similar movies using cosine similarity.
   - For a Letterboxd user, the system fetches watched films, matches them to the dataset, and recommends similar movies.

3. **API:**  
   - `/recommendations/`: Get recommendations for a specific movie title.
   - `/letterboxd/`: Get recommendations based on a Letterboxd user's watched films.

---

## File Overview

- **vectorization.py**  
  Cleans and vectorizes the TMDB dataset, saving vectors and metadata.

- **cosine_similarity.py**  
  Loads vectors and metadata, and provides recommendation functions for both movie titles and Letterboxd users.

- **main.py**  
  FastAPI app exposing the recommendation endpoints.

- **helper_functions.py**  
  Utility functions for cleaning titles, handling Roman numerals, and TMDB API interaction.

- **data/**  
  Contains the TMDB dataset and generated vector/metadata files.

---

## Quickstart

### 1. Install Requirements

```
pip install -r requirements.txt
```
or (if using pyproject.toml):
```
uv pip sync pyproject.toml
```

### 2. Prepare Data

- Place your TMDB movie metadata CSV in `data/`.
- Run `vectorization.py` to generate `tfidf_matrix.npz` and `movie_metadata.parquet`.

### 3. Set TMDB API Key

Create a `.env` file in the project root:
```
TMDB_API_KEY=your_tmdb_api_key_here
```

### 4. Run the API

```
uvicorn main:app --reload
```
http://127.0.0.1:8000/docs


### 5. Example API Usage

- **Get recommendations for a movie:**
  ```
  GET /recommendations/?movie_title=Inception&num_recommendations=5
  ```

- **Get recommendations for a Letterboxd user:**
  ```
  GET /letterboxd/?user_name=your_letterboxd_username&num_recommendations=5
  ```

---
- **Example Output:**
{
  "recommendations": [
    [
      "Da 5 Bloods",
      [
        {
          "id": 349547,
          "title": "Lai Taihan",
          "similarity": 0.8051160984772696
        },
        {
          "id": 925389,
          "title": "Thành phố thất lậc",
          "similarity": 0.7552430554518925
        },
        {
          "id": 424899,
          "title": "Vietnam",
          "similarity": 0.7361414630824842
        }
      ]
    ],
    [
      "Rambo III",
      [
        {
          "id": 344551,
          "title": "Afghantsi",
          "similarity": 0.5091628961119101
        },
        {
          "id": 558903,
          "title": "Охотники за караванами",
          "similarity": 0.4425797684560486
        },
        {
          "id": 1208716,
          "title": "Barefoot",
          "similarity": 0.4206878035860692
        }
      ]
    ],
    [
      "Casino",
      [
        {
          "id": 1241843,
          "title": "The Vortex",
          "similarity": 0.4426637774463832
        },
        {
          "id": 566518,
          "title": "Cashed Out Casino",
          "similarity": 0.3941176648334844
        },
        {
          "id": 80751,
          "title": "Blackjack",
          "similarity": 0.3928062778871885
        }
      ]
    ]
  ]
}

## Example Python Usage

```python
from cosine_similarity import get_recommendations, letter_boxd_get_recommendations

print(get_recommendations("Inception", num_recommendations=5))
print(letter_boxd_get_recommendations("your_letterboxd_username", num_recommendations=5))
```

---

## Requirements

- Python 3.12+
- pandas
- scikit-learn
- scipy
- requests
- beautifulsoup4
- fastapi
- uvicorn
- tmdbv3api
- python-dotenv
- pyarrow

---

## Notes

- Ensure your TMDB API key is set in `.env`.
- The system expects cleaned and matched movie titles between Letterboxd and TMDB.
- For best results, use the provided helper functions for title normalization.

---
## Potential Future Improvement
- Larger scale with a user authentication and rating system
- Above would lead to more options of how to adjust the vectorization and cosine similarity to improve quality of recommendations