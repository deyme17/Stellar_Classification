import pickle
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from app.utils.dataloader import DataLoader 
from app.settings.constants import VAL_CSV, SAVED_ESTIMATOR


with open('app/settings/specifications.json') as f:
    specifications = json.load(f)

x_columns = specifications['description']['X']
y_column = specifications['description']['y']

raw_val = pd.read_csv(VAL_CSV)
x_raw = raw_val[x_columns]

loader = DataLoader()
loader.fit(x_raw)
X = loader.load_data()
y = raw_val['class']

loaded_model = pickle.load(open(SAVED_ESTIMATOR, 'rb'))
print(loaded_model.score(X, y))