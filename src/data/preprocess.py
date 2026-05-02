from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_data(
    ratings_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Load, clean, and split ratings data.
    No manual encoding is required for Surprise SVD.
    """

    print(f"Loading dataset: {ratings_path}")

    # Load dataset
    df = pd.read_csv(ratings_path)

    # Standardize column names
    df = df.rename(columns={
        "userId": "user_id",
        "movieId": "movie_id",
    })

    # Validate schema
    required_columns = ["user_id", "movie_id", "rating"]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Clean data
    df = df.drop_duplicates()
    df = df.dropna(subset=required_columns)

    # Train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    # Save datasets
    train_df.to_parquet(
        ARTIFACTS_DIR / "train_data.parquet",
        index=False,
    )

    test_df.to_parquet(
        ARTIFACTS_DIR / "test_data.parquet",
        index=False,
    )

    # Save metadata
    metadata = {
        "num_users": train_df["user_id"].nunique(),
        "num_movies": train_df["movie_id"].nunique(),
        "num_ratings": len(train_df),
    }

    joblib.dump(
        metadata,
        ARTIFACTS_DIR / "dataset_metadata.pkl"
    )

    print("\nArtifacts created successfully:")
    print(" - train_data.parquet")
    print(" - test_data.parquet")
    print(" - dataset_metadata.pkl")

    print("\nColumns:")
    print(train_df.columns.tolist())

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    preprocess_data("data/ratings.csv")