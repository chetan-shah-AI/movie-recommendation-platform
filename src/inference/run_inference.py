from src.utils.model_loader import ModelLoader
from src.inference.recommender import MovieRecommender


def run_single_case(recommender, case: dict):
    print("\n" + "=" * 80)
    print(case["description"])
    print("=" * 80)

    try:
        recommendations = recommender.recommend(
            user_id=case["user_id"],
            top_n=case.get("top_n", 5),
            genre_filter=case.get("genre_filter"),
            min_predicted_score=case.get("min_predicted_score", 0.0),
        )

        if not recommendations:
            print("No recommendations found.")
        else:
            for rec in recommendations:
                print(rec)

    except Exception as e:
        print(f"Error during inference: {e}")


def main():
    print("\nLoading model and artifacts...")

    loader = ModelLoader(
        artifacts_dir="artifacts",
        movies_path="data/movies.csv",
    )

    assets = loader.load()

    print("Model and data loaded successfully.")

    recommender = MovieRecommender(
        model=assets["model"],
        ratings_df=assets["train_data"],
        movies_df=assets["movies"],
    )

    test_cases = [
        # {
        #     "description": "Known user — Top 10 recommendations",
        #     "user_id": 161935,
        #     "top_n": 5,
        #     "genre_filter": "Drama",
        #     "min_predicted_score": 4.5,
        # },
        {
            "description": "Unknown user — cold start fallback",
            "user_id": 99999999999999,
            "top_n": 10,
            "genre_filter": "Drama",
            "min_predicted_score": 4.5,
        },
    ]

    for case in test_cases:
        run_single_case(recommender, case)


if __name__ == "__main__":
    main()