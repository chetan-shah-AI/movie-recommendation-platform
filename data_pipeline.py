import data_ingestion as di
import config

# Dataset paths
movies_file = config.MOVIES_DATASET_PATH
ratings_file = config.RATINGS_DATASET_PATH

# Load data only a subset for testing
movies_data = di.get_movies_data(movies_file).head(1000)
ratings_data = di.get_ratings_data(ratings_file).head(100000)


# movies_data.head(1000)
# ratings_data.head(10000)


# print("Movies data:", movies_data.head())
# print("Ratings data:", ratings_data.head())


def create_user_movie_matrix(ratings=ratings_data):
    
    user_movie_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    return user_movie_matrix

user_movie_matrix = create_user_movie_matrix()


print("User-Movie Matrix:", user_movie_matrix)


