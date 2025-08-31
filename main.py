""" 
FastAPI application for movie recommendations using cosine similarity and Letterboxd user data.
"""

from fastapi import FastAPI, Query, HTTPException


from cosine_similarity import get_recommendations, letter_boxd_get_recommendations


app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "API for movie recommendations based on cosine similarity "
        "and Letterboxd user data."
    )
)

@app.get("/recommendations/")
def recommend(movie_title: str = Query(..., description="Movie Title"),
              num_recommendations: int = 10):
    """
    Recommend movies based on a given movie title using cosine similarity.
    
    Args:
        movie_title (str): The title of the movie to base recommendations on.
        num_recommendations (int): Number of recommendations to return (default: 10).
    
    Returns:
        dict: A dictionary containing a list of recommended movies.
    """
    try:
        recs = get_recommendations(movie_title, num_recommendations)
        return {"recommendations": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

@app.get("/letterboxd/")
def letterboxd_recommendations(
    user_name: str = Query(..., description="Letterboxd Username"),
    num_recommendations: int = 10,
    recommendations_per_film: int = 5
):
    """
    Recommend movies for a Letterboxd user based on their watched films and cosine similarity.

    Args:
        user_name (str): Letterboxd username to base recommendations on.
        num_recommendations (int): Number of recommendations to return (default: 10).
        recommendations_per_film (int): Number of recommendations per watched film (default: 5).

    Returns:
        dict: A dictionary containing a list of recommended movies.
    """
    try:
        recs = letter_boxd_get_recommendations(
            user_name,
            num_recommendations,
            recommendations_per_film=recommendations_per_film
        )
        return {"recommendations": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
