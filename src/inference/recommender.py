import pandas as pd
from typing import List, Dict, Optional

from src.utils.model_loader import ModelLoader
from src.inference.cold_start import popularity_recommendations


class MovieRecommender:
    """
    Main inference engine for movie recommendations.

    Supports:
    - Personalized recommendations
    - Cold-start fallback
    - Genre filtering
    - Minimum predicted score filtering
    - Top-N ranking
    """

    def __init__(
        self,
        model,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
    ):
        """
        Initialize recommender with model and datasets.
        """

        # -----------------------------------------
        # Standardize column names
        # -----------------------------------------
        self.ratings_df = ratings_df.rename(
            columns={
                "movieId": "movie_id",
                "userId": "user_id",
            }
        )

        self.movies_df = movies_df.rename(
            columns={
                "movieId": "movie_id",
            }
        )

        self.model = model

        # -----------------------------------------
        # Cached sets for fast lookup
        # -----------------------------------------
        self.all_movie_ids = set(
            self.movies_df["movie_id"].unique()
        )

        self.all_user_ids = set(
            self.ratings_df["user_id"].unique()
        )

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _get_unseen_movies(
        self,
        user_id: int,
    ) -> List[int]:
        """
        Return movies the user has not rated yet.
        """

        watched_movies = set(
            self.ratings_df[
                self.ratings_df["user_id"] == user_id
            ]["movie_id"]
        )

        unseen_movies = list(
            self.all_movie_ids - watched_movies
        )

        return unseen_movies

    def _predict_ratings(
        self,
        user_id: int,
        unseen_movies: List[int],
    ) -> pd.DataFrame:
        """
        Predict ratings for unseen movies.
        """

        predictions = []

        for movie_id in unseen_movies:

            try:
                prediction = self.model.predict(
                    user_id,
                    movie_id,
                )

                estimated_rating = getattr(
                    prediction,
                    "est",
                    None,
                )

                if estimated_rating is not None:

                    predictions.append(
                        {
                            "movie_id": movie_id,
                            "predicted_rating": round(
                                estimated_rating,
                                3,
                            ),
                        }
                    )

            except Exception:
                # Skip failed predictions safely
                continue

        if not predictions:
            return pd.DataFrame()

        return pd.DataFrame(predictions)

    def _merge_movie_metadata(
        self,
        predictions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach movie titles and genres.
        """

        merged_df = predictions_df.merge(
            self.movies_df,
            on="movie_id",
            how="left",
        )

        return merged_df

    def _apply_filters(
        self,
        recommendations_df: pd.DataFrame,
        genre_filter: Optional[str],
        min_predicted_score: float,
    ) -> pd.DataFrame:
        """
        Apply recommendation filtering logic.
        """

        # -----------------------------------------
        # Genre filtering
        # -----------------------------------------
        if genre_filter:

            recommendations_df = recommendations_df[
                recommendations_df["genres"].str.contains(
                    genre_filter,
                    case=False,
                    na=False,
                )
            ]

        # -----------------------------------------
        # Minimum predicted score filtering
        # -----------------------------------------
        recommendations_df = recommendations_df[
            recommendations_df["predicted_rating"]
            >= min_predicted_score
        ]

        return recommendations_df

    def _rank_results(
        self,
        recommendations_df: pd.DataFrame,
        top_n: int,
    ) -> pd.DataFrame:
        """
        Rank recommendations by predicted rating.
        """

        ranked_df = recommendations_df.sort_values(
            by="predicted_rating",
            ascending=False,
        ).head(top_n)

        return ranked_df

    # =========================================================
    # MAIN PUBLIC METHOD
    # =========================================================

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        genre_filter: Optional[str] = None,
        min_predicted_score: float = 0.0,
    ) -> List[Dict]:
        """
        Generate recommendations for a user.
        """

        # =====================================================
        # COLD START
        # =====================================================

        if user_id not in self.all_user_ids:
            print(
                f"User ID {user_id} not found. Using cold-start fallback."
            )

            cold_start_results = popularity_recommendations(
                ratings_df=self.ratings_df,
                movies_df=self.movies_df,
                top_n=top_n,
            )

            return cold_start_results.to_dict(
                orient="records"
            )

        # =====================================================
        # PERSONALIZED RECOMMENDATIONS
        # =====================================================

        unseen_movies = self._get_unseen_movies(
            user_id=user_id,
        )

        predictions_df = self._predict_ratings(
            user_id=user_id,
            unseen_movies=unseen_movies,
        )

        if predictions_df.empty:
            return []

        recommendations_df = self._merge_movie_metadata(
            predictions_df=predictions_df,
        )

        recommendations_df = self._apply_filters(
            recommendations_df=recommendations_df,
            genre_filter=genre_filter,
            min_predicted_score=min_predicted_score,
        )

        recommendations_df = self._rank_results(
            recommendations_df=recommendations_df,
            top_n=top_n,
        )

        # =====================================================
        # RETURN CLEAN OUTPUT
        # =====================================================

        output_columns = [
            "movie_id",
            "title",
            "genres",
            "predicted_rating",
        ]

        if "release_year" in recommendations_df.columns:
            output_columns.append("release_year")

        recommendations_df = recommendations_df[
            output_columns
        ]

        return recommendations_df.to_dict(
            orient="records"
        )


# =============================================================
# LOCAL TESTING
# =============================================================

if __name__ == "__main__":

    print(
        "Loading recommender system..."
    )

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

    # =========================================================
    # TEST CASES
    # =========================================================

    test_cases = [
        # {
        #     "description": "Known user - Top 5",
        #     "user_id": 4,
        #     "top_n": 5,
        #     "genre_filter": None,
        #     "min_predicted_score": 0.0,
        # },
        {
            "description": "Sci-Fi only",
            "user_id": 4,
            "top_n": 5,
            "genre_filter": "Sci-Fi",
            "min_predicted_score": 4.0,
        },
        # {
        #     "description": "Drama only with minimum score",
        #     "user_id": 4,
        #     "top_n": 5,
        #     "genre_filter": "Drama",
        #     "min_predicted_score": 3.5,
        # },
        # {
        #     "description": "Cold-start unknown user",
        #     "user_id": 9999,
        #     "top_n": 5,
        #     "genre_filter": None,
        #     "min_predicted_score": 0.0,
        # },
    ]

    # =========================================================
    # RUN TESTS
    # =========================================================

    for case in test_cases:

        print("\n" + "=" * 80)
        print(case["description"])
        print("=" * 80)

        recommendations = recommender.recommend(
            user_id=case["user_id"],
            top_n=case["top_n"],
            genre_filter=case["genre_filter"],
            min_predicted_score=case[
                "min_predicted_score"
            ],
        )

        if not recommendations:
            print("No recommendations found.")
            continue

        for recommendation in recommendations:
            print(recommendation)

