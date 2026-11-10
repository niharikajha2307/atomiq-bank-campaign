import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from src.preprocessing import build_quantum_preprocessor

class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """
    Quantum Kernel Classifier implemented with Qiskit circuits.
    
    Architecture:
    1. ZZFeatureMap encodes continuous classical features x into 2^k dimensional quantum Hilbert space:
       - Hadamard gates create equal superposition.
       - Rz gates encode first-order feature rotations.
       - CNOT-Rz-CNOT sequences encode pairwise ZZ interactions between adjacent qubits.
    2. Quantum Fidelity Kernel:
       - Computes K(x, y) = |<psi(y)|psi(x)>|^2 using exact Qiskit Statevector simulation.
    3. Classification & Calibration:
       - Fits an SVC with precomputed kernel.
       - Utilizes Platt scaling (probability=True) to produce continuous probabilities in [0, 1].
    """
    def __init__(self, n_qubits=5, reps=1, C=1.0, class_weight='balanced', random_state=42):
        self.n_qubits = n_qubits
        self.reps = reps
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement='linear')
        self.svc_ = None
        self.X_train_statevectors_ = None

    def _compute_statevectors(self, X):
        statevectors = []
        for x in X:
            # Bind parameters to the Qiskit ZZFeatureMap circuit
            bound_circuit = self.feature_map.assign_parameters(x)
            sv = Statevector.from_instruction(bound_circuit)
            statevectors.append(sv.data)
        return np.array(statevectors)

    def fit(self, X, y):
        # Convert X to array if DataFrame
        X_arr = np.asarray(X, dtype=float)
        self.X_train_statevectors_ = self._compute_statevectors(X_arr)
        
        # Compute Gram matrix: K_train = |V_train @ V_train^H|^2
        K_train = np.abs(self.X_train_statevectors_ @ self.X_train_statevectors_.conj().T) ** 2
        
        # Fit SVC with Platt scaling for probability calibration
        self.svc_ = SVC(
            kernel='precomputed',
            C=self.C,
            probability=True,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.svc_.fit(K_train, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_test_sv = self._compute_statevectors(X_arr)
        # Compute test-vs-train kernel matrix: K_test = |V_test @ V_train^H|^2
        K_test = np.abs(X_test_sv @ self.X_train_statevectors_.conj().T) ** 2
        return self.svc_.predict_proba(K_test)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_test_sv = self._compute_statevectors(X_arr)
        K_test = np.abs(X_test_sv @ self.X_train_statevectors_.conj().T) ** 2
        return self.svc_.predict(K_test)

def get_quantum_models(random_state=42):
    """
    Returns quantum model pipelines with preprocessing and feature reduction included.
    """
    models = {}
    
    # 1. 5-Qubit Quantum Kernel Classifier with linear ZZ entanglement
    qk_svc_5 = QuantumKernelSVC(n_qubits=5, reps=1, C=1.0, class_weight='balanced', random_state=random_state)
    models['QuantumKernel_5Qubit'] = Pipeline([
        ('preprocessor', build_quantum_preprocessor(k_features=5)),
        ('classifier', qk_svc_5)
    ])
    
    # 2. 4-Qubit Quantum Kernel Classifier (more compact Hilbert space)
    qk_svc_4 = QuantumKernelSVC(n_qubits=4, reps=1, C=1.0, class_weight='balanced', random_state=random_state)
    models['QuantumKernel_4Qubit'] = Pipeline([
        ('preprocessor', build_quantum_preprocessor(k_features=4)),
        ('classifier', qk_svc_4)
    ])
    
    return models
