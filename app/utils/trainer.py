from sklearn.ensemble import GradientBoostingClassifier


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return GradientBoostingClassifier(n_estimators=185, learning_rate=0.6, 
                                          max_depth=11, random_state=17, 
                                          min_samples_split=4, max_features=None
                                          ).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)