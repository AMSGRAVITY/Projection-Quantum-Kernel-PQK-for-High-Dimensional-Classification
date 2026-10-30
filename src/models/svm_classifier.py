from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from src.kernels.projected_kernel import ProjectedQuantumKernel

class QuantumKernelClassifier:

    def __init__(
            self,
            kernel_engine: ProjectedQuantumKernel,
            C: float = 1.0
    ):
        self.kernel_engine = kernel_engine
        self.C = C
        self.svm = SVC(kernel="precomputed", C=self.C)
        self.train_features: np.ndarray = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "QuantumKernelClassifier":
        self.train_features = self.kernel_engine.extract_features(X_train)
        # Gram Matrix (Train x Train)
        K_train = self.kernel_engine.compute_gram_matrix(self.train_features)

        self.svm.fit(K_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:

        if self.train_features is None:
            raise ValueError("Model isn't fit properly !!!")

        test_features = self.kernel_engine.extract_features(X_test)
        # Gram Matrix (Test x Train)
        K_test = self.kernel_engine.compute_gram_matrix(test_features, self.train_features)

        return self.svm.predict(K_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str]:
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return acc, report