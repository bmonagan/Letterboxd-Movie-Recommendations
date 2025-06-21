from fastapi import FastAPI, HTTPException, Query


from cosine_similarity import get_recommendations

app = FastAPI()

@app.get("/recommendations/")
def recommend(movie_title: str = Query(..., description="Movie Title"), num_recommendations: int = 10):
    try:
        recs = get_recommendations(movie_title, num_recommendations)
        return {"recommendations": recs}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))