import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # feature selection
        Xcols = ['u', 'g', 'r', 'i', 'z', 'redshift', 'plate']
        self.dataset = self.dataset[Xcols]

        # missing data
        self.dataset['plate'] = self.dataset['plate'].fillna(0)
        if self.dataset.isnull().sum().any():
            self.dataset = self.dataset.fillna(self.dataset.mean())

        # outliars
            # 1) IQR
        outliar_cols = ['u', 'g', 'r', 'i', 'z']
        for column in outliar_cols:
            Q1 = self.dataset[column].quantile(0.25)
            Q3 = self.dataset[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            self.dataset.loc[self.dataset[column] < lower_limit, column] = lower_limit
            self.dataset.loc[self.dataset[column] > upper_limit, column] = upper_limit
            # 2) log tranformation
        self.dataset['redshift'] = np.log1p(self.dataset['redshift'])

        # label encoding
        le_plate = LabelEncoder()
        self.dataset['plate'] = le_plate.fit_transform(self.dataset['plate'])

        # standartization
        ss = StandardScaler()
        cont_cols = Xcols.copy()
        cont_cols.remove('plate')
        self.dataset[cont_cols] = ss.fit_transform(self.dataset[cont_cols])

        return self.dataset
