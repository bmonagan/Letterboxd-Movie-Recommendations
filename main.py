from fastapi import FastAPI, HTTPException, Query


from cosine_similarity import get_recommendations, letter_boxd_get_recommendations


app = FastAPI(title = "Movie Recommendation API", description = "API for movie recommendations based on cosine similarity and Letterboxd user data.")

@app.get("/recommendations/")
def recommend(movie_title: str = Query(..., description="Movie Title"), num_recommendations: int = 10):
    try:
        recs = get_recommendations(movie_title, num_recommendations)
        return {"recommendations": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/letterboxd/")
def letterboxd_recommendations(user_name: str = Query(..., description="Letterboxd Username"), num_recommendations: int = 10):
    try:
        recs = letter_boxd_get_recommendations(user_name, num_recommendations)
        return {"recommendations": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))