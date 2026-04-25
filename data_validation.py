import pandas as pd
from data_ingestion import get_ratings_data, get_movies_data
import config

# o user_id (required)
# o movie_id (required)
# o rating (required)
# o timestamp (optional)

required_columns = ["userId", "movieId", "rating"]


def validate_schema(data, required_columns):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    print("Schema validation passed.")
    return True


ratings_file = config.RATINGS_DATASET_PATH
ratings_data = get_ratings_data(ratings_file)


validate_schema(ratings_data, required_columns)

