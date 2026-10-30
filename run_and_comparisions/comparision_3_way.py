"""
Executes the comprehensive 3-way validation:
  1. Global Fidelity Kernel (Baseline)
  2. Projected Quantum Kernel (Aer Simulated NISQ Noise)
  3. Projected Quantum Kernel (Physical IBM QPU)
"""

import numpy as np
from src.circuits.feature_map import QuantumFeatureMapBuilder
from src.kernels.fidelity_kernel import FidelityQuantumKernel
from src.kernels.projected_kernel import ProjectedQuantumKernel
from src.noise.noise_models import build_nisq_noise_model
from src.hardware.ibm_executor import IBMHardwareExecutor
from src.data.data_loader import get_scaled_data

if __name__ == "__main__":
    n_qubits = 4
    n_samples = 6  # Small compact batch: preserves monthly quota & runs quickly in queue

    print(f"Loading {n_samples} benchmark points (Qubits: {n_qubits})...")
    X_val, _, y_val, _ = get_scaled_data(
        dataset_name="breast_cancer",
        n_qubits=n_qubits,
        n_samples=n_samples,
        random_state=42,
    )

    # 1. Build circuit & local observables
    builder = QuantumFeatureMapBuilder(num_qubits=n_qubits, reps=1)
    circuit = builder.build_circuit()
    observables = builder.get_local_pauli_observables()

    print("\n=======================================================")
    print("      STEP 1: BASELINE GLOBAL FIDELITY KERNEL         ")
    print("=======================================================")
    fid_engine = FidelityQuantumKernel(circuit=circuit)
    K_fidelity = fid_engine.compute_gram_matrix(X_val)
    print("Global Fidelity Gram Matrix (Sample 3x3 subset):\n", np.round(K_fidelity[:3, :3], 4))

    print("\n=======================================================")
    print("      STEP 2: PROJECTED KERNEL (SIMULATED NISQ NOISE) ")
    print("=======================================================")
    noise_model = build_nisq_noise_model(p_1gate=0.001, p_2gate=0.015, p_readout=0.02)
    pqk_sim = ProjectedQuantumKernel(
        circuit=circuit,
        observables=observables,
        gamma=0.5,
        backend=noise_model,
        shots=2000,
    )
    feats_sim = pqk_sim.extract_features(X_val)
    K_sim = pqk_sim.compute_gram_matrix(feats_sim)
    print("Simulated Noise Gram Matrix (Sample 3x3 subset):\n", np.round(K_sim[:3, :3], 4))

    print("\n=======================================================")
    print("      STEP 3: PROJECTED KERNEL (REAL IBM QPU)         ")
    print("=======================================================")
    hw_executor = IBMHardwareExecutor()
    isa_circuit, isa_observables = hw_executor.transpile_circuit_and_observables(
        circuit, observables, optimization_level=3
    )
    feats_qpu, job_id = hw_executor.run_hardware_batch(
        isa_circuit, isa_observables, X_val, shots=2000
    )
    K_qpu = pqk_sim.compute_gram_matrix(feats_qpu)
    print("Real Hardware Gram Matrix (Sample 3x3 subset):\n", np.round(K_qpu[:3, :3], 4))

    print("\n=======================================================")
    print("               QUANTITATIVE HARDWARE COMPARISON        ")
    print("=======================================================")
    # Calculate Mean Absolute Error (MAE) between Simulated Noise and Real Hardware Gram matrices
    mae_sim_vs_qpu = np.mean(np.abs(K_sim - K_qpu))
    mae_fid_vs_qpu = np.mean(np.abs(K_fidelity - K_qpu))

    print(f"IBM Job ID:                                {job_id}")
    print(f"Gram Matrix MAE (Simulated Noise vs QPU):  {mae_sim_vs_qpu:.4f}")
    print(f"Gram Matrix MAE (Global Fidelity vs QPU):  {mae_fid_vs_qpu:.4f}")
    print("=======================================================")