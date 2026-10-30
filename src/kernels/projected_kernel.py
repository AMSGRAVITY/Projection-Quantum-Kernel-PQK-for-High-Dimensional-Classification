from typing import List, Optional
import numpy as np
from scipy.spatial.distance import cdist
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp

class ProjectedQuantumKernel:

    def __init__(
            self,
            circuit: QuantumCircuit,
            observables: List[SparsePauliOp],
            gamma: float = 1.0,
            backend: Optional[NoiseModel] = None,
            shots: Optional[int] = 4000,
    ):
        self.circuit = circuit
        self.observables = observables
        self.gamma = gamma
        self.shots = shots
        self.backend = backend

        if self.backend is None:
            self.estimator = StatevectorEstimator()
            self.is_state_vector = True
        else:
            self.estimator = AerEstimator()
            self.estimator.options.backend_options = {"noise_model": self.backend}
            self.is_state_vector = False

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        n_obs = len(self.observables)
        projected_features = np.zeros((n_samples, n_obs), dtype=np.float64)

        # Build PUBs (Primitive Unified Blocs)
        if self.is_state_vector:
            pubs = [(self.circuit, self.observables, x) for x in X]
        else:
            pubs = [(self.circuit, self.observables, x, 1.0 / np.sqrt(self.shots)) for x in X]

        job = self.estimator.run(pubs)
        results = job.result()

        for i, pub_result in enumerate(results):
            projected_features[i] = np.asarray(pub_result.data.evs, dtype=np.float64)

        return projected_features


    # Computes the RBF Gram Matrix
    # K(a, b) = exp(-gamma * ||a - b||^2)
    def compute_gram_matrix(
            self,
            features_a: np.ndarray,
            features_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        is_kernel = self.is_state_vector is None

        if features_b is None:
            features_b = features_a

        # We find ||a - b||^2
        sq_dist = cdist(features_a, features_b, metric="sqeuclidean")

        K = np.exp(-self.gamma * sq_dist)

        if is_kernel:
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)

        return K