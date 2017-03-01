import numpy as np
import pandas as pd


class LeastSquaresClassification:  # Linear Regression

    def __init__(self):
        self._coefficient = []

    def fit(self, x, y):
        x_full = np.column_stack((np.ones(x.shape[0]), x))
        self._coefficient = np.dot(np.linalg.pinv(x_full), y)

    def predict(self, x):
        x_full = np.column_stack((np.ones(x.shape[0]), x))
        y = np.dot(x_full, self._coefficient)
        y[y <= 0.5] = 0
        y[y > 0.5] = 1
        return y


class NearestNeighborClassification:  # kNN

    def __init__(self, k):
        self._k = k
        self._coefficient = []

    def fit(self, x, y):
        self._x = x
        self._y = y

    def predict(self, x):
        y = np.zeros((x.shape[0], 1))
        length = np.zeros((self._x.shape[0], 2))
        length[:, 1] = self._y
        import datetime
        now = datetime.datetime.now()
        for i in range(x.shape[0]):
            length[:, 0] = (self._x[:, 0] - x[i, 0])**2 + (self._x[:, 1] - x[i, 1])**2
            # y[i] = pd.Series(length[:, 0]).nsmallest(self._k).mean()  # this Windows version is too old
            y[i] = np.sum(length[length[:, 0].argsort()][0:self._k, 1])/self._k
        print(datetime.datetime.now() - now)
        y[y <= 0.5] = 0
        y[y > 0.5] = 1
        return y