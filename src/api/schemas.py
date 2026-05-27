from typing import Optional
from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5
    genre_filter: Optional[str] = None
    min_predicted_score: float = 0.0
    include_explanation: bool = False