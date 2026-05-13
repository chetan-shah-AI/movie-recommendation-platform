from pydantic import BaseModel
from typing import Optional, List


class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5
    genre_filter: Optional[str] = None
    min_predicted_score: Optional[float] = 0.0


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    release_year: int
    predicted_rating: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendation_type: str
    recommendations: List[RecommendationItem]