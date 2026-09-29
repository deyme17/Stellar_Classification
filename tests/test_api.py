import json
import pandas as pd
import requests

from app.utils import DataLoader
from app.settings.constants import VAL_CSV


def test_api(val_path: str) -> None:
    with open("app/settings/specifications.json") as f:
        specifications = json.load(f)

    info = specifications["description"]
    x_columns = info["X"]
    y_column = info["y"]
    metrics = info["metrics"]

    val_set = pd.read_csv(val_path, header=0)
    val_x = val_set[x_columns]
    val_y = val_set[y_column]

    loader = DataLoader()
    loader.fit(val_x)
    val_processed = loader.load_data()
    print("data:", val_processed[:10])

    req_data = {"data": json.dumps(val_x.to_dict())}

    response = requests.get(
        "http://127.0.0.1:8000/predict",
        data=req_data,
    )

    api_predict = response.json()["prediction"]
    print("predict:", api_predict[:10])

    api_score = eval(metrics)(val_y, api_predict)
    print("accuracy:", api_score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-path", default=VAL_CSV, help="Path to validation dataset")
    args = parser.parse_args()

    test_api(args.val_path)