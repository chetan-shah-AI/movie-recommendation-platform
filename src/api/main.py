from fastapi import FastAPI, HTTPException

from src.api.schemas import RecommendationRequest, RecommendationResponse
from src.utils.model_loader import ModelLoader
from src.inference.recommender import MovieRecommender
from src.ai.explanation_service import ExplanationService


app = FastAPI(
    title="Movie Recommendation API",
    description="API for serving personalized, cold-start, and LLM-explained movie recommendations.",
    version="1.1.0",
)


recommender = None
explanation_service = None


@app.on_event("startup")
def load_services():
    global recommender
    global explanation_service

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

        explanation_service = ExplanationService()

        print("Recommender and explanation service loaded successfully.")

    except Exception as e:
        print(f"Failed to load services: {e}")
        recommender = None
        explanation_service = None


@app.get("/health")
def health_check():
    if recommender is None:
        return {
            "status": "unhealthy",
            "message": "Recommender not loaded",
        }

    return {
        "status": "healthy",
        "message": "Movie Recommendation API is running",
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_movies(request: RecommendationRequest):
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

        explanation = None

        if request.include_explanation:
            if explanation_service is None:
                raise HTTPException(
                    status_code=503,
                    detail="Explanation service is not available.",
                )

            explanation = explanation_service.generate_explanation(
                user_id=request.user_id,
                recommendations_df=recommendations_df,
                recommendation_type=recommendation_type,
                genre_filter=request.genre_filter,
            )

        

        recommendations_df = recommendations_df.fillna("")
        recommendations = recommendations_df.to_dict(orient="records")


        return {
            "user_id": request.user_id,
            "recommendation_type": recommendation_type,
            "recommendations": recommendations,
            "explanation": explanation,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}",
        )