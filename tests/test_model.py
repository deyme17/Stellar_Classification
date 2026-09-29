import argparse
import json
import pickle
import pandas as pd

from app.utils.dataloader import DataLoader
from app.settings.constants import VAL_CSV, SAVED_ESTIMATOR


def test_model(val_path: str, model_path: str) -> None:
    with open("app/settings/specifications.json") as f:
        specifications = json.load(f)

    info = specifications["description"]
    x_columns = info["X"]
    y_column = info["y"]

    raw_val = pd.read_csv(val_path)
    x_raw = raw_val[x_columns]
    y = raw_val[y_column]

    loader = DataLoader()
    loader.fit(x_raw)
    X = loader.load_data()

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    print("accuracy:", loaded_model.score(X, y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", default=VAL_CSV, help="Path to validation dataset")
    parser.add_argument("--model-path", default=SAVED_ESTIMATOR, help="Path to saved estimator")
    args = parser.parse_args()

    test_model(args.val_path, args.model_path)