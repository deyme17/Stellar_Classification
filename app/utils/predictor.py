import os
import pickle
import pandas as pd
import numpy as np
from app.settings.constants import SAVED_ESTIMATOR


class Predictor:
    def __init__(self):
        if not os.path.exists(SAVED_ESTIMATOR):
            raise FileNotFoundError(f"Model file not found: {SAVED_ESTIMATOR}")
        
        with open(SAVED_ESTIMATOR, 'rb') as f:
            self.loaded_estimator = pickle.load(f)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.loaded_estimator.predict(data)