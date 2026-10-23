from typing import Tuple, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumFeatureMapBuilder:
    # We construct entangling circuits and local Pauli observables for projected quantum kernel (PQK)

    def __init__(self, num_qubits: int, reps: int = 2):
        self.num_qubits = num_qubits
        self.reps = reps
        self.parameters = ParameterVector("x", length=num_qubits)