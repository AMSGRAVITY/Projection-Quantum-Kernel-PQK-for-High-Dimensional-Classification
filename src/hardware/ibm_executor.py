from typing import List, Optional
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as IBMEstimator

class IBMHardwareExecutor:
    # Real Quantum Computer Execution (IBM)

    def __init__(
            self,
            backend_name: Optional[str] = None
    ):

        self.service = QiskitRuntimeService(channel="ibm_quantum_platform", instance="auto")

        if backend_name:
            self.backend = self.service.backend(backend_name)
        else:
            print("Selecting Least Busy Physical QPU...")
            self.backend = self.service.least_busy(operational=True, simulator=False)

        print(f"Targeting Physical Backend: {self.backend.name}")
        print(f"Total Physical Qubits: {self.backend.num_qubits}")

    def transpile_circuit_and_observables(
            self,
            circuit: QuantumCircuit,
            observables: List[SparsePauliOp],
            optimization_level: int = 3,
    ) -> tuple[QuantumCircuit, List[SparsePauliOp]]:
        # Transforms abstract circuits and observables into hardware-nate ISA format

        print(f"Transpiling circuit for {self.backend.name} (Optimization Level: {optimization_level})...")

        isa_circuit = transpile(
            circuits=circuit,
            backend=self.backend,
            optimization_level=optimization_level,
            seed_transpiler=42
        )

        isa_observables = [obs.apply_layout(isa_circuit.layout) for obs in observables]

        print(f" -> Transpiled Circuit Depth: {isa_circuit.depth()}")
        print(f" -> Physical Ops Count: {isa_circuit.count_ops()}")
        return isa_circuit, isa_observables

    def run_hardware_batch(
            self,
            isa_circuit: QuantumCircuit,
            isa_observables: List[SparsePauliOp],
            X_batch: np.ndarray,
            shots: int = 4000,
    ) -> tuple[np.ndarray, str]:

        estimator = IBMEstimator(mode=self.backend)
        estimator.options.default_shots = shots

        pubs = [(isa_circuit, isa_observables, x) for x in X_batch]

        print(f"Submitting batch of {len(X_batch)} circuits to {self.backend.name}...")
        job = estimator.run(pubs)
        job_id = job.job_id()
        print(f"Job Succesfully queries! Job ID: {job_id}")

        results = job.result()
        print("Hardware execution completed!")

        projected_features = np.zeros((len(X_batch), len(isa_observables)), dtype=np.float64)
        for i, pub_res in enumerate(results):
            projected_features[i] = np.asarray(pub_res.data.evs, dtype=np.float64)

        return projected_features, job_id