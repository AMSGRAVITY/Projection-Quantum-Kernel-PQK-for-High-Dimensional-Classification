from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

def build_nisq_noise_model(
        p_1gate: float = 0.001,
        p_2gate: float = 0.015,
        p_readout: float = 0.02
) -> NoiseModel:

    noise_model = NoiseModel()

    # For 1 qubit gates (H, Rz, Ry, X, Sx)
    error_1q = depolarizing_error(p_1gate, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["h", "rz", "ry", "x", "sx"])

    # For 2 qubit gates (CNOTS)
    error_2q = depolarizing_error(p_2gate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])

    # Measurement Error
    ro_prob = [[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]
    error_ro = ReadoutError(ro_prob)
    noise_model.add_all_qubit_readout_error(error_ro)

    return noise_model