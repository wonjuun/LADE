"""kNN-based discrimination against the harmful reference set (Eq. 5)."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def featurize(P, w):
    X = P * w
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


class KNNDetector:
    def __init__(self, R, rho=0.9, K=5, n_boot=200, seed=42):
        self.knn = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(R)
        d = self.knn.kneighbors(R, n_neighbors=K + 1)[0][:, 1:].mean(1)
        rng = np.random.RandomState(seed)
        boot = [np.quantile(d[rng.randint(0, len(d), len(d))], rho) for _ in range(n_boot)]
        self.tau = float(np.median(boot))

    def distance(self, X):
        return self.knn.kneighbors(X)[0].mean(1)

    def __call__(self, X):
        return self.distance(X) <= self.tau
