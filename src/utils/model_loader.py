import joblib
from pathlib import Path


class ModelLoader:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

    def load(self):
        model = joblib.load(self.artifacts_dir / "svd_model.pkl")
        movies = joblib.load(self.artifacts_dir / "movies.pkl")
        ratings = joblib.load(self.artifacts_dir / "ratings.pkl")

        return {
            "model": model,
            "movies": movies,
            "ratings": ratings,
        }

if __name__ == "__main__":
    loader = ModelLoader()
    artifacts = loader.load()
    print("Artifacts loaded successfully:")
    print(f"- Model: {artifacts['model']}")
    print(f"- Movies: {artifacts['movies'].shape}")
    print(f"- Ratings: {artifacts['ratings'].shape}")