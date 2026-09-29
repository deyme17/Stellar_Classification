import pandas as pd


class Dataset:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self._df = pd.read_csv(csv_file)

    def len(self) -> int:
        """
        Get number of examples
        @return: int
        """
        return len(self._df)

    def columns(self) -> list:
        """
        Get list of columns names
        @return: list
        """
        return list(self._df.columns)

    def getitem(self, index: int) -> tuple[list, int]:
        """
        Get example by index
        @param index: int
        @return: list, int
        """
        'Generates one sample of data'
        if index < 0 or index >= len(self._df):
            raise IndexError("Index out of range")
    
        row = self._df.iloc[index]
        X = row.drop(columns=['class']).tolist()
        y = row['class']

        return X, y

    def get_items(self, items_number: int) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get specific amount of examples
        @param items_number: int
        @return: pd.DataFrame, pd.Series
        """
        if items_number < 0 or items_number >= len(self._df):
            raise IndexError("Index out of range")
        
        rows = self._df.iloc[:items_number]
        Xs = rows.drop('class', axis=1)
        ys = rows['class']

        return Xs, ys
