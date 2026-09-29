import argparse

import pandas as pd

from app.utils.dataloader import DataLoader
from app.settings.constants import TRAIN_CSV


def test_load(path: str) -> None:
    df = pd.read_csv(path, header=0)
    loader = DataLoader()
    loader.fit(df)
    print(loader.load_data())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path", default=TRAIN_CSV, help="Dataset path")
    args = parser.parse_args()

    test_load(args.path)