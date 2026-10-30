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

    def build_circuit(self) -> QuantumCircuit:
        # we build a hard-ware efficient parameterized entangled Quantum circuit here
        qc = QuantumCircuit(self.num_qubits, name="PQK_FeatureMap")

        # Using H for superpositon (ket(+))
        for q in range(self.num_qubits):
            qc.h(q)

        for _ in range(self.reps):
            for q in range(self.num_qubits):
                qc.rz(self.parameters[q], q)
                qc.ry(self.parameters[q], q)

            for q in range(self.num_qubits - 1):
                qc.cx(q, q+1)
            if self.num_qubits > 2:
                qc.cx(self.num_qubits-1, 0)

        return qc

    def get_local_pauli_observables(self) -> List[SparsePauliOp]:
        observables = []
        pauli_bases = ["X", "Y", "Z"]

        for q in range(self.num_qubits):
            for pauli in pauli_bases:
                op_list = ["I"] * self.num_qubits
                op_list[self.num_qubits - 1 - q] = pauli
                pauli_str = "".join(op_list)
                observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

        return observables