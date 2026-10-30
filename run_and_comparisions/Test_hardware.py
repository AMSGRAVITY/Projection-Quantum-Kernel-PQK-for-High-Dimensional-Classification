"""
Tests the hardware pipeline on IBM Quantum QPUs.
"""

import numpy as np
from src.circuits.feature_map import QuantumFeatureMapBuilder
from src.data.data_loader import get_scaled_data
from src.hardware.ibm_executor import IBMHardwareExecutor

if __name__ == "__main__":
    n_qubits = 4
    n_test_samples = 4

    print("Loading test data...")
    X_train, _, _, _ = get_scaled_data(
        dataset_name="breast_cancer",
        n_qubits=n_qubits,
        n_samples=n_test_samples,
        random_state=42,
    )

    # 1. Build circuit and local Pauli observables
    builder = QuantumFeatureMapBuilder(num_qubits=n_qubits, reps=1)
    circuit = builder.build_circuit()
    observables = builder.get_local_pauli_observables()

    # 2. Connect to IBM Quantum
    hw_executor = IBMHardwareExecutor()

    # 3. Transpile to hardware ISA
    isa_circuit, isa_observables = hw_executor.transpile_circuit_and_observables(
        circuit, observables, optimization_level=3
    )

    # 4. Execute on the real QPU
    features_qpu, job_id = hw_executor.run_hardware_batch(
        isa_circuit, isa_observables, X_train, shots=2000
    )

    print("\n================ HARDWARE RUN SUMMARY ================")
    print(f"IBM Job ID:              {job_id}")
    print(f"Extracted Feature Shape: {features_qpu.shape}")
    print("Sample QPU Expectation Values (Point 0):\n", np.round(features_qpu[0], 4))
    print("======================================================")