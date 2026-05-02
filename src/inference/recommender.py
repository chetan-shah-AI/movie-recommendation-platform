import pandas as pd
from typing import List, Dict

from src.inference.cold_start import popularity_recommendations


class MovieRecommender:
    def __init__(self, model, ratings_df, movies_df):
        self.model = model
        self.ratings_df = ratings_df
        self.movies_df = movies_df

        self.all_movie_ids = set(movies_df["movie_id"].unique())
        self.all_user_ids = set(ratings_df["user_id"].unique())

    def _get_unseen_movies(self, user_id: int) -> List[int]:
        watched = set(
            self.ratings_df[
                self.ratings_df["user_id"] == user_id
            ]["movie_id"]
        )

        return list(self.all_movie_ids - watched)

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        min_pred_rating: float = 0.0,
    ) -> List[Dict]:
        """Generate Top-N recommendations."""

        # Cold-start handling
        if user_id not in self.all_user_ids:
            return (
                popularity_recommendations(
                    self.ratings_df,
                    self.movies_df,
                    top_n=top_n,
                )
                .to_dict(orient="records")
            )

        unseen_movies = self._get_unseen_movies(user_id)

        predictions = []

        for movie_id in unseen_movies:
            pred = self.model.predict(user_id, movie_id)

            if pred.est >= min_pred_rating:
                predictions.append(
                    {
                        "movie_id": movie_id,
                        "predicted_rating": round(pred.est, 3),
                    }
                )

        if not predictions:
            return []

        pred_df = pd.DataFrame(predictions)

        pred_df = pred_df.sort_values(
            "predicted_rating",
            ascending=False,
        ).head(top_n)

        results = pred_df.merge(
            self.movies_df,
            on="movie_id",
            how="left",
        )

        return results.to_dict(orient="records")