import pickle
from settings.constants import SAVED_ESTIMATOR
import os


class Predictor:
    def __init__(self):
        if not os.path.exists(SAVED_ESTIMATOR):
            raise FileNotFoundError(f"Model file not found: {SAVED_ESTIMATOR}")
        
        with open(SAVED_ESTIMATOR, 'rb') as f:
            self.loaded_estimator = pickle.load(f)

    def predict(self, data):
        return self.loaded_estimator.predict(data)