from src.utils.model_loader import ModelLoader
from src.inference.recommender import MovieRecommender


def run_single_case(recommender, case: dict):
    print("\n" + "=" * 80)
    print(case["description"])
    print("=" * 80)

    # try:
    recommendations = recommender.recommend(
        user_id=case["user_id"],
        top_n=case.get("top_n", 5),
        # genre_filter=case.get("genre_filter"),
        # min_predicted_score=case.get("min_predicted_score"),
    )

    # if recommendations.empty:
    #     print("No recommendations found.")
    # else:
    print(recommendations)

    # except Exception as e:
    #     print(f"Error during inference: {e}")


def main():
    print("\nLoading model and artifacts...")

    loader = ModelLoader(
        artifacts_dir="artifacts",
        movies_path="data/movies.csv",
    )

    assets = loader.load()

    print("Model and data loaded successfully.")

    # ⚠️ IMPORTANT:
    # Set this based on how you trained the model
    # False → trained using user_id, movie_id
    # True  → trained using user_idx, item_idx
    USE_ENCODED_IDS = False

    recommender = MovieRecommender(
    model=assets["model"],
    ratings_df=assets["train_data"],
    movies_df=assets["movies"],
)

    # 🔥 Test scenarios
    test_cases = [
        {
            "description": "Known user — Top 5 recommendations",
            "user_id": 1001,
            "top_n": 10,
        },
        {
            "description": "Known user — Sci-Fi filter",
            "user_id": 1001,
            "top_n": 10,
        },
        {
            "description": "Known user — Drama, score >= 3.5",
            "user_id": 1001,
            "top_n": 10,
        },
        {
            "description": "Unknown user — cold start fallback",
            "user_id": 9999,
            "top_n": 10,
        },
    ]

    for case in test_cases:
        run_single_case(recommender, case)


if __name__ == "__main__":
    main()


    