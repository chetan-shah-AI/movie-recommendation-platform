import joblib
import pandas as pd
from pathlib import Path


class ModelLoader:
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        movies_path: str = "data/movies.csv",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.movies_path = Path(movies_path)

    def load(self):
        model_path = self.artifacts_dir / "model.pkl"
        train_data_path = self.artifacts_dir / "train_data.parquet"
        test_data_path = self.artifacts_dir / "test_data.parquet"
        metadata_path = self.artifacts_dir / "dataset_metadata.pkl"

        model = joblib.load(model_path)
        train_data = pd.read_parquet(train_data_path)
        test_data = pd.read_parquet(test_data_path)
        metadata = joblib.load(metadata_path)
        movies = pd.read_csv(self.movies_path)

        return {
            "model": model,
            "train_data": train_data,
            "test_data": test_data,
            "metadata": metadata,
            "movies": movies,
        }

if __name__ == "__main__":
    loader = ModelLoader()
    artifacts = loader.load()
    print("Artifacts loaded successfully:")
    print(f"- Model: {artifacts['model']}")
    print(f"- Movies: {artifacts['movies'].shape}")
    print(f"- Ratings: {artifacts['ratings'].shape}")