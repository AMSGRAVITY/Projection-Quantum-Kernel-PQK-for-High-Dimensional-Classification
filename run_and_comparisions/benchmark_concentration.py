"""
Demonstrates the curse of dimensionality / Barren Plateau problem in standard kernels
vs. the stability of Projected Quantum Kernels as qubits scale.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.circuits.feature_map import QuantumFeatureMapBuilder
from src.kernels.projected_kernel import ProjectedQuantumKernel
from src.kernels.fidelity_kernel import FidelityQuantumKernel
from src.data.data_loader import get_scaled_data


def compute_off_diagonal_variance(K: np.ndarray) -> float:
    """
    Computes the variance of off-diagonal elements in a Gram matrix.
    If variance -> 0, the kernel is collapsing into the Identity matrix.
    """
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diagonals = K[mask]
    return float(np.var(off_diagonals))


if __name__ == "__main__":
    qubit_range = [4, 6, 8, 10, 12, 14, 16]
    fidelity_variances = []
    projected_variances = []

    print("==================================================")
    print("    RUNNING KERNEL CONCENTRATION BENCHMARK        ")
    print("==================================================")

    for n_q in qubit_range:
        print(f"\n---> Evaluating n_qubits = {n_q}...")
        X_train, _, _, _ = get_scaled_data(
            dataset_name="breast_cancer",
            n_qubits=n_q,
            n_samples=30,  # Fast check
            random_state=42,
        )

        builder = QuantumFeatureMapBuilder(num_qubits=n_q, reps=2)
        circuit = builder.build_circuit()
        observables = builder.get_local_pauli_observables()

        # 1. Global Fidelity Kernel
        fid_kernel = FidelityQuantumKernel(circuit=circuit)
        K_fid = fid_kernel.compute_gram_matrix(X_train)
        var_fid = compute_off_diagonal_variance(K_fid)
        fidelity_variances.append(var_fid)

        # 2. Projected Quantum Kernel
        pqk = ProjectedQuantumKernel(circuit=circuit, observables=observables, gamma=1.0 / n_q)
        feats = pqk.extract_features(X_train)
        K_pqk = pqk.compute_gram_matrix(feats)
        var_pqk = compute_off_diagonal_variance(K_pqk)
        projected_variances.append(var_pqk)

        print(f"     Global Fidelity Off-Diag Variance: {var_fid:.6e}")
        print(f"     Projected Kernel Off-Diag Variance: {var_pqk:.6e}")

    print("\n================ FINAL COMPARISON ================")
    print(f"{'Qubits':<8} | {'Fidelity Kernel Var':<22} | {'Projected Kernel Var':<22}")
    print("-" * 58)
    for q, vf, vp in zip(qubit_range, fidelity_variances, projected_variances):
        print(f"{q:<8} | {vf:<22.6e} | {vp:<22.6e}")
    print("==================================================")