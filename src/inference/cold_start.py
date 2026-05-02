import pandas as pd


def popularity_recommendations(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_n: int = 10,
    min_ratings: int = 50,
) -> pd.DataFrame:
    """
    Return popularity-based recommendations
    for cold-start users.
    """

    stats = (
        ratings_df.groupby("movie_id")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count"),
        )
        .reset_index()
    )

    stats = stats[
        stats["rating_count"] >= min_ratings
    ]

    ranked = stats.sort_values(
        ["avg_rating", "rating_count"],
        ascending=False,
    )

    recommendations = ranked.merge(
        movies_df,
        on="movie_id",
        how="left",
    )

    return recommendations.head(top_n)


if __name__ == "__main__":
    # Load datasets
    ratings_df = pd.read_parquet(
        "artifacts/train_data.parquet"
    )

    movies_df = pd.read_csv(
        "data/movies.csv"
    )

    # Standardize movie column name
    movies_df = movies_df.rename(
        columns={"movieId": "movie_id"}
    )

    # Generate recommendations
    recommendations = popularity_recommendations(
        ratings_df=ratings_df,
        movies_df=movies_df,
        top_n=10,
        min_ratings=50,
    )

    print("\nTop Popular Movies:\n")
    print(
        recommendations[
            [
                "movie_id",
                "title",
                "genres",
                "avg_rating",
                "rating_count",
            ]
        ].to_string(index=False)
    )