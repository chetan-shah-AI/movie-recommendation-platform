from typing import Optional

import pandas as pd
from openai import OpenAI
from langfuse import observe, get_client

from src.config.settings import settings
from src.ai.prompt_templates import (
    MOVIE_EXPLANATION_SYSTEM_PROMPT,
    MOVIE_EXPLANATION_USER_PROMPT,
)


class ExplanationService:
    """
    Generates natural-language explanations for movie recommendations.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL

    @observe(name="generate_recommendation_explanation")
    def generate_explanation(
        self,
        user_id: int,
        recommendations_df: pd.DataFrame,
        recommendation_type: str,
        genre_filter: Optional[str] = None,
    ) -> str:
        """
        Generate an explanation for recommendation results.

        Args:
            user_id: User receiving recommendations.
            recommendations_df: DataFrame returned by recommender.
            recommendation_type: personalized or cold_start.
            genre_filter: Optional genre filter.

        Returns:
            Human-readable explanation string.
        """

        if recommendations_df.empty:
            return "No recommendations were available for this request."

        movies_text = self._format_movies_for_prompt(recommendations_df)

        user_prompt = MOVIE_EXPLANATION_USER_PROMPT.format(
            recommendation_type=recommendation_type,
            user_id=user_id,
            movies=movies_text,
        )

        if genre_filter:
            user_prompt += f"\nApplied genre filter: {genre_filter}"

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": MOVIE_EXPLANATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        explanation = response.output_text

        langfuse = get_client()
        trace_url = langfuse.get_trace_url()

        print(f"Langfuse trace URL: {trace_url}")

        return explanation

    def _format_movies_for_prompt(self, recommendations_df: pd.DataFrame) -> str:
        """
        Convert recommended movies dataframe into compact prompt text.
        """

        rows = []

        for _, row in recommendations_df.iterrows():
            rows.append(
                f"- {row['title']} "
                f"({row['release_year']}), "
                f"genres: {row['genres']}, "
                f"predicted rating: {row['predicted_rating']}"
            )

        return "\n".join(rows)