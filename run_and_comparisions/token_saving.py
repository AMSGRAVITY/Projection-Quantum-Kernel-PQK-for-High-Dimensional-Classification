# Save IBM Token here...

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_TOKEN",
    overwrite=True,
    set_as_default=True
)

# once saved, you don't need to write it again...