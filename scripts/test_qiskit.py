from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create a 2-qubit Bell state circuit (entangled state (|00> + |11>) / sqrt(2))
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Run on AerSimulator
simulator = AerSimulator()
result = simulator.run(qc, shots=1000, seed_simulator=42).result()
counts = result.get_counts()

print("Qiskit test successful!")
print("Measured counts on AerSimulator (Bell State):", counts)
