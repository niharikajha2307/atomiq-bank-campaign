import time
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler

# Test 4 qubits on 10 samples
X = np.random.RandomState(42).randn(10, 4)
feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement='linear')

# Method A: Qiskit FidelityQuantumKernel
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
K_qiskit = kernel.evaluate(x_vec=X, y_vec=X)

# Method B: Statevector batching
statevectors = []
for x in X:
    bound_circuit = feature_map.assign_parameters(x)
    sv = Statevector.from_instruction(bound_circuit)
    statevectors.append(sv.data)
V = np.array(statevectors)  # shape (10, 16)
# Inner product matrix
inner_products = np.abs(V @ V.conj().T) ** 2

max_diff = np.max(np.abs(K_qiskit - inner_products))
print(f"Max absolute difference between Qiskit Kernel and Statevector Inner Product: {max_diff:.2e}")
