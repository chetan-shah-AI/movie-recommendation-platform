import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


RAW_RATINGS_PATH = "data/ratings.csv"

ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DIR = Path("data/processed")

USER_ENCODER_PATH = ARTIFACTS_DIR / "user_encoder.pkl"
ITEM_ENCODER_PATH = ARTIFACTS_DIR / "item_encoder.pkl"
TRAIN_DATA_PATH = ARTIFACTS_DIR / "train_data.parquet"

# TRAIN_DATA_PATH = PROCESSED_DIR / "train_data.parquet"
# TEST_DATA_PATH = PROCESSED_DIR / "test_data.parquet"


def main():
    print("Creating artifacts...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load ratings data
    df = pd.read_csv(RAW_RATINGS_PATH)

    # 2. Clean basic issues
    df = df.dropna(subset=["userId", "movieId", "rating"])
    df = df.drop_duplicates(subset=["userId", "movieId"])
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]

    # 3. Create encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_idx"] = user_encoder.fit_transform(df["userId"])
    df["item_idx"] = item_encoder.fit_transform(df["movieId"])

    # 4. Split train/test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    # 5. Save encoders
    joblib.dump(user_encoder, USER_ENCODER_PATH)
    joblib.dump(item_encoder, ITEM_ENCODER_PATH)

    # 6. Save cleaned training data
    train_df.to_parquet(TRAIN_DATA_PATH, index=False)
    # test_df.to_parquet(TEST_DATA_PATH, index=False)

    print("Artifacts created successfully:")
    print(f"- {USER_ENCODER_PATH}")
    print(f"- {ITEM_ENCODER_PATH}")
    print(f"- {TRAIN_DATA_PATH}")
    # print(f"- {TEST_DATA_PATH}")

def create_movies_ratings_artifacts():

    movies_df = pd.read_csv("data/movies.csv")
    ratings_df = pd.read_csv("data/ratings.csv")

    joblib.dump(movies_df, "artifacts/movies.pkl")
    joblib.dump(ratings_df, "artifacts/ratings.pkl")





if __name__ == "__main__":
    main()
    create_movies_ratings_artifacts()