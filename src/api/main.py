import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import RecommendationRequest
from src.utils.model_loader import ModelLoader
from src.inference.recommender import MovieRecommender
from src.ai.explanation_service import ExplanationService


app = FastAPI(
    title="Movie Recommendation API",
    description="Movie recommender with OpenAI explanations and Langfuse tracing.",
    version="1.2.0",
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

        print("Recommender, OpenAI explanation service, and Langfuse loaded.")

    except Exception as e:
        print(f"Startup failed: {e}")
        recommender = None
        explanation_service = None


@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API is running",
        "llm_explanations": "enabled",
        "observability": "langfuse",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if recommender is not None else "unhealthy",
        "recommender_loaded": recommender is not None,
        "explanation_service_loaded": explanation_service is not None,
    }


@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender service is not available.",
        )

    try:
        raw_recommendations = recommender.recommend(
            user_id=request.user_id,
            top_n=request.top_n,
            genre_filter=request.genre_filter,
            min_predicted_score=request.min_predicted_score,
        )

        if raw_recommendations is None:
            recommendations_df = pd.DataFrame()
        elif isinstance(raw_recommendations, pd.DataFrame):
            recommendations_df = raw_recommendations.copy()
        elif isinstance(raw_recommendations, list):
            recommendations_df = pd.DataFrame(raw_recommendations)
        else:
            raise TypeError(
                f"Unsupported recommender output type: {type(raw_recommendations)}"
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

        clean_df = recommendations_df.fillna("")
        recommendations = clean_df.to_dict(orient="records")

        return {
            "user_id": request.user_id,
            "recommendation_type": recommendation_type,
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
            "explanation": explanation,
            "observability": {
                "provider": "langfuse",
                "llm_tracing_enabled": request.include_explanation,
            },
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"Recommendation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}",
        )