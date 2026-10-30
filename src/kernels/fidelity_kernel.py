from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer import AerSimulator

class FidelityQuantumKernel:
    # Computes Standard Global State Fidelity Kernels via Circuit inversion

    def __init__(
            self,
            circuit: QuantumCircuit,
            backend: Optional[AerSimulator] = None,
            shots: Optional[int] = 4000,
    ):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.num_qubits = circuit.num_qubits

        if self.backend is None:
            self.sampler = StatevectorSampler()
            self.is_state_vector = True
        else:
            self.sampler = AerSimulator(run_options={"shots": self.shots})
            self.is_state_vector = False

    def _build_overlap_circuit(self) -> QuantumCircuit:
        n = self.num_qubits
        qc = QuantumCircuit(n, n) # n qubits and n classical bits

        param_a = self.circuit.parameters
        param_b = [f"b_{i}" for i in range(len(param_a))]

        qc.compose(self.circuit, inplace=True)
        inv_circuit = self.circuit.inverse()

        param_dict = {p: qiskit_param for p, qiskit_param in zip(inv_circuit.parameters, param_b)}
        inv_circuit.assign_parameters(param_dict, inplace=True)

        qc.compose(inv_circuit, inplace=True)
        qc.measure(range(n), range(n))
        return qc

    def compute_gram_matrix(self, X_a: np.ndarray, X_b: Optional[np.ndarray] = None) -> np.ndarray:
        is_self = X_b is None

        if is_self:
            X_b = X_a

        sv_a = [Statevector(self.circuit.assign_parameters(xa)) for xa in X_a]
        sv_b = sv_a if is_self else [Statevector(self.circuit.assign_parameters(xb)) for xb in X_b]

        n_a, n_b = len(X_a), len(X_b)
        K = np.zeros((n_a, n_b))

        for i in range(n_a):
            for j in range(n_b):
                if is_self and i == j:
                    K[i, j] = 1.0
                elif is_self and j < i:
                    K[i, j] = K[j, i]
                else:
                    fidelity = np.abs(sv_a[i].inner(sv_b[j])) ** 2
                    K[i, j] = fidelity

        return K