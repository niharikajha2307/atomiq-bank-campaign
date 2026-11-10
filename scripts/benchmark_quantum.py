import time
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler

# Let's test 4 features on 40 samples
X = np.random.RandomState(42).randn(40, 4)
feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement='linear')

sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

t0 = time.time()
K = kernel.evaluate(x_vec=X[:20], y_vec=X[:20])
t1 = time.time()

print(f"20x20 StatevectorSampler Kernel Matrix calculated in {t1 - t0:.2f} seconds.")
print("Kernel shape:", K.shape)
print("Diagonal mean (should be 1.0):", np.mean(np.diag(K)))
