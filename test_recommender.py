import pandas as pd
import joblib

from src.inference.recommender import MovieRecommender

# Load artifacts
model = joblib.load("artifacts/svd_model.pkl")
ratings_df = pd.read_csv("data/raw/ratings_200.csv")
movies_df = pd.read_csv("data/raw/movies.csv")

# Create recommender
recommender = MovieRecommender(
    model=model,
    ratings_df=ratings_df,
    movies_df=movies_df,
)

# Test known user
print("\n=== Known User ===")
results = recommender.recommend(user_id=1001, top_n=5)
for r in results:
    print(r)

# Test unknown user (cold start)
print("\n=== Unknown User ===")
results = recommender.recommend(user_id=9999, top_n=5)
for r in results:
    print(r)