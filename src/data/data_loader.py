from typing import Tuple
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_scaled_data(
        dataset_name: str = "breast_cancer",
        n_qubits: int = 4,
        n_samples: int = 80,
        test_size: float = 0.3,
        random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target

        if len(X) > n_samples:
            indices = np.random.RandomState(random_state).choice(len(X), n_samples, replace=False)
            X, y = X[indices], y[indices]
    elif dataset_name == "synthetic":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=max(n_qubits, 4),
            n_informative=max(2, n_qubits - 1),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown Dataset: {dataset_name}")

    if X.shape[1] != n_qubits:
        pca = PCA(n_components=n_qubits, random_state=random_state)
        X = pca.fit_transform(X)

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)