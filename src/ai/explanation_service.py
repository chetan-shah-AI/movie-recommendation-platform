import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langfuse import observe, get_client
from langfuse.openai import openai

from src.ai.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.config.settings import settings


load_dotenv()


class ExplanationService:
    """
    Uses OpenAI/ChatGPT-style LLM calls to explain movie recommendations.
    Langfuse traces the explanation workflow and model call.
    """

    def __init__(self):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

        self.client = openai.OpenAI()
        self.model = settings.LLM_MODEL

    @observe(name="movie_recommendation_explanation")
    def generate_explanation(
        self,
        user_id: int,
        recommendations_df,
        recommendation_type: str,
        genre_filter: Optional[str] = None,
    ) -> str:
        """
        Generate an LLM explanation for recommended movies.
        """

        recommendations_df = self._ensure_dataframe(recommendations_df)

        if recommendations_df.empty:
            return "No recommendations were available to explain."

        movies_text = self._format_movies(recommendations_df)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            user_id=user_id,
            recommendation_type=recommendation_type,
            genre_filter=genre_filter or "None",
            movies_text=movies_text,
        )

        langfuse = get_client()

        with langfuse.start_as_current_observation(
            as_type="generation",
            name="openai_movie_explanation",
            model=self.model,
        ) as generation:

            generation.update(
                input={
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "user_id": user_id,
                    "recommendation_type": recommendation_type,
                    "genre_filter": genre_filter,
                }
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.3,
            )

            explanation = response.choices[0].message.content

            generation.update(
                output={
                    "explanation": explanation,
                }
            )

        langfuse.flush()

        return explanation

    def _ensure_dataframe(self, recommendations_df) -> pd.DataFrame:
        """
        Accept either DataFrame or list and normalize to DataFrame.
        """

        if recommendations_df is None:
            return pd.DataFrame()

        if isinstance(recommendations_df, pd.DataFrame):
            return recommendations_df.copy()

        if isinstance(recommendations_df, list):
            return pd.DataFrame(recommendations_df)

        raise TypeError(
            f"Unsupported recommendations type: {type(recommendations_df)}"
        )

    def _format_movies(self, recommendations_df: pd.DataFrame) -> str:
        """
        Convert recommendation rows into compact LLM prompt text.
        """

        rows = []

        for _, row in recommendations_df.iterrows():
            title = row.get("title", "Unknown title")
            genres = row.get("genres", "Unknown genre")
            release_year = row.get("release_year", "Unknown year")
            predicted_rating = row.get("predicted_rating", "Unknown score")

            rows.append(
                f"- {title} ({release_year}) | "
                f"Genres: {genres} | "
                f"Predicted rating: {predicted_rating}"
            )

        return "\n".join(rows)