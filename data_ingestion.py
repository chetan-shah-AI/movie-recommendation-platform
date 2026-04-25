import pandas as pd

def get_ratings_data(ratings_file):
    ratings = pd.read_csv(ratings_file)
    return ratings


def get_movies_data(movies_file):
    movies = pd.read_csv(movies_file)
    return movies





