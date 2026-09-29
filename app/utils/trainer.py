import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class Estimator:
    @staticmethod
    def fit(train_x: np.ndarray, train_y: np.ndarray) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=185, learning_rate=0.6, random_state=17,
            max_depth=11, min_samples_split=4, max_features=None,
        ).fit(train_x, train_y)

    @staticmethod
    def predict(trained: GradientBoostingClassifier, 
                test_x: np.ndarray) -> np.ndarray:
        return trained.predict(test_x)