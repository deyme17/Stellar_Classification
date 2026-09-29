from app.utils.dataset import Dataset


def test_dataset(path: str) -> None:
    ds = Dataset(path)

    print("1. len test")
    print(f"{ds.len()}\n")

    print("2. columns test")
    print(f"{ds.columns()}\n")

    print("3. getitem test")
    print(f"{ds.getitem(2)}\n")

    print("4. getitems test")
    print(f"{ds.get_items(5)}")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Dataset path")
    args = parser.parse_args()

    test_dataset(args.path)