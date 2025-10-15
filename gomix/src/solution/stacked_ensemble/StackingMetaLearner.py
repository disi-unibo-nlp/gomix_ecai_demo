from sklearn.linear_model import LinearRegression
import numpy as np


class StackingMetaLearner:
    def __init__(self, n_classes: int):
        if n_classes < 1:
            raise ValueError('n_classes must be a positive integer.')
        self._n_classes = n_classes
        self._regressions = None

    def fit(self, base_scores: np.ndarray, labels: np.ndarray):
        """
        Base scores: (n_train_samples, n_classes, n_base_models)
        Labels: (n_train_samples, n_classes)
        """
        assert base_scores.ndim == 3 and labels.ndim == 2 \
               and base_scores.shape[0] == labels.shape[0] \
               and base_scores.shape[1] == self._n_classes \
               and labels.shape[1] == self._n_classes

        # Coefs may differ between classes.
        self._regressions = [LinearRegression(fit_intercept=False) for _ in range(self._n_classes)]

        for i in range(self._n_classes):
            self._regressions[i].fit(base_scores[:, i, :], labels[:, i])

    def predict(self, base_scores: np.ndarray) -> np.ndarray:
        assert self._n_classes is not None and isinstance(self._regressions, list)

        # Base scores: (n_samples, n_classes, n_base_models)
        assert base_scores.ndim == 3 and base_scores.shape[1] == self._n_classes

        # Predictions: (n_samples, n_classes)
        predictions = np.zeros((base_scores.shape[0], self._n_classes))
        for i in range(self._n_classes):
            predictions[:, i] = self._regressions[i].predict(base_scores[:, i, :])
        return predictions
