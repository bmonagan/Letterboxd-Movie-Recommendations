from fastapi import FastAPI, HTTPException, Query


from cosine_api_testing import get_recommendations

app = FastAPI()

@app.get("/recommendations/")
def recommend(movie_id: int = Query(..., description="TMDB Movie ID"), num_recommendations: int = 5):
    try:
        recs = get_recommendations(movie_id, num_recommendations)
        return {"recommendations": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))