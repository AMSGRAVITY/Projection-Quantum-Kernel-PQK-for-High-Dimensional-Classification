"""
Plots:
  1. Exponential Concentration Scaling (n = 4 to 16 qubits)
  2. Side-by-Side 3-Way Gram Matrix Heatmaps (Global Fidelity vs Aer vs Real IBM QPU)
"""

# Here I did with already obtained points from the previous run, you can directly take those points and plot it after the run...

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# PLOT 1: Concentration Scaling (n = 4 to 16 qubits)
# -------------------------------------------------------------
qubits = [4, 6, 8, 10, 12, 14, 16]
fidelity_var = [3.69e-2, 2.10e-2, 7.14e-3, 1.90e-3, 3.27e-4, 4.67e-5, 6.98e-6]
projected_var = [1.28e-2, 5.61e-3, 3.73e-3, 2.77e-3, 3.02e-3, 2.35e-3, 2.07e-3]

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
ax.plot(qubits, fidelity_var, marker='o', color='#d62828', linewidth=2.2, label=r'Global Fidelity Kernel $\vert{}\langle\psi(x)\vert{}\psi(x^\prime)\rangle\vert{}^2$')
ax.plot(qubits, projected_var, marker='s', color='#003049', linewidth=2.2, label=r'Projected Quantum Kernel (PQK $\rho_k$)')

ax.set_yscale('log')
ax.set_xlabel('Qubit Count ($n$)', fontsize=11, fontweight='bold')
ax.set_ylabel('Off-Diagonal Kernel Variance (Log Scale)', fontsize=11, fontweight='bold')
ax.set_title('Exponential Kernel Concentration vs. PQK Dimensional Stability', fontsize=12, fontweight='bold')
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('kernel_concentration_scaling.png')
plt.close()
print("Saved 'kernel_concentration_scaling.png' successfully!")

# -------------------------------------------------------------
# PLOT 2: 3-Way Hardware & Simulation Gram Matrix Heatmaps
# -------------------------------------------------------------
K_fidelity = np.array([
    [1.0000, 0.2246, 0.2736],
    [0.2246, 1.0000, 0.4343],
    [0.2736, 0.4343, 1.0000]
])

K_sim = np.array([
    [1.0000, 0.5266, 0.4444],
    [0.5266, 1.0000, 0.4173],
    [0.4444, 0.4173, 1.0000]
])

K_qpu = np.array([
    [1.0000, 0.5747, 0.4990],
    [0.5747, 1.0000, 0.5428],
    [0.4990, 0.5428, 1.0000]
])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

matrices = [K_fidelity, K_sim, K_qpu]
titles = [
    "Global Fidelity Kernel\n(Ideal Baseline)",
    "Projected Kernel (PQK)\n(Simulated NISQ Noise)",
    "Projected Kernel (PQK)\n(Physical IBM QPU: ibm_fez)"
]

for idx, (ax, mat, title) in enumerate(zip(axes, matrices, titles)):
    cax = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["$x_0$", "$x_1$", "$x_2$"])
    ax.set_yticklabels(["$x_0$", "$x_1$", "$x_2$"])
    
    # Annotate numeric values
    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight="bold")

fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.8, label="Kernel Similarity Metric $K(x_i, x_j)$")
plt.savefig('hardware_gram_comparison.png', bbox_inches='tight')
plt.close()
print("Saved 'hardware_gram_comparison.png' successfully!")