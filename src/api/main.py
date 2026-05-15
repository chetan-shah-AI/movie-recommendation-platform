from fastapi import FastAPI, HTTPException
from src.api.schemas import RecommendationRequest, RecommendationResponse
from src.utils.model_loader import ModelLoader
from src.inference.recommender import MovieRecommender


app = FastAPI(
    title="Movie Recommendation API",
    description="API for serving personalized and cold-start movie recommendations.",
    version="1.0.0",
)


recommender = None


@app.on_event("startup")
def load_recommender():
    """
    Load model and data once when the API starts.
    """

    global recommender

    try:
        loader = ModelLoader(
            artifacts_dir="artifacts",
            movies_path="data/movies.csv",
        )

        assets = loader.load()

        recommender = MovieRecommender(
            model=assets["model"],
            ratings_df=assets["train_data"],
            movies_df=assets["movies"],
        )

        print("Recommender loaded successfully.")

    except Exception as e:
        print(f"Failed to load recommender: {e}")
        recommender = None


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    if recommender is None:
        return {
            "status": "unhealthy",
            "message": "Recommender not loaded",
        }

    return {
        "status": "healthy",
        "message": "Recommender API is running",
    }


@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    """
    Generate movie recommendations.
    """

    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender service is not available.",
        )

    try:
        recommendations_df = recommender.recommend(
            user_id=request.user_id,
            top_n=request.top_n,
            genre_filter=request.genre_filter,
            min_predicted_score=request.min_predicted_score,
        )

        recommendation_type = (
            "cold_start"
            if request.user_id not in recommender.all_user_ids
            else "personalized"
        )
        print(recommendations_df)

        # recommendations = recommendations_df.to_dict(orient="records")

        return {
            "user_id": request.user_id,
            "recommendation_type": recommendation_type,
            "recommendations": recommendations_df,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}",
        )