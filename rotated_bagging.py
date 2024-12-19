from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
from utilities import random_orthogonal_matrix

class RotatedIsolationForest(BaseEstimator, OutlierMixin):
    """ Isolation Forest с механизмом Rotation Bagging. """
    def __init__(self, num_base_estimators=10, **kwargs):
        self.num_base_estimators = num_base_estimators
        self.estimators = []
        for _ in range(self.num_base_estimators):
            self.estimators.append(IsolationForest(**kwargs))

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        num_features = X.shape[1]
        self.rotations = [np.eye(num_features)] # Первого обучаем на "сырых" данных
        self.rotations += [random_orthogonal_matrix(num_features) for _ in range(self.num_base_estimators - 1)]
        for i in range(self.num_base_estimators):
            data_X = X @ self.rotations[i]
            self.estimators[i].fit(data_X)

        return self

    def predict(self, X):
        predictions = []
        for rotation, estimator in zip(self.rotations, self.estimators):
            data_X = X @ rotation
            predictions.append(estimator.predict(data_X))
        return np.mean(predictions, axis=0) > 0.5

    def score_samples(self, X):
        check_is_fitted(self)
        scoring = []
        for rotation, estimator in zip(self.rotations, self.estimators):
            data_X = X @ rotation
            scoring.append(estimator.score_samples(data_X))
        return np.mean(scoring, axis=0)

    def decision_function(self, X):
        # Возвращаем отрицание, чтобы быть совместимым с scikit-learn API
        return self.score_samples(X)