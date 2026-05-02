import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle


def train_svd_model(ratings):

    # Define rating scale
    reader = Reader(rating_scale=(0.5, 5.0))

    # Load data into Surprise
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']],
        reader
    )


    trainset, testset = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )


    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )


    model.fit(trainset)

    return model, testset


def evaluate_model(model, testset):
    
    predictions = model.test(testset)
    mse = accuracy.mse(predictions)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print(f"Evaluation Metrics:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")

    return mse, rmse, mae

def save_model(model, path="artifacts/movie_recommendation.pkl"):

    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="artifacts/movie_recommendation.pkl"):
    
    with open(path, "rb") as f:
        model = pickle.load(f)

    return model