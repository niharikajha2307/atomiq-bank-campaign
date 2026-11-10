import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.quantum_models import get_quantum_models

df_train = pd.read_csv("data/Stage1_Train.csv", dtype=str, keep_default_na=False)
X = df_train.drop(columns=['customer_id', 'y'])
y = (df_train['y'] == 'yes').astype(int)

models = get_quantum_models(random_state=42)
pipeline = models['QuantumKernel_4Qubit']

print("Fitting QuantumKernel_4Qubit on 50 samples...")
pipeline.fit(X.iloc[:50], y.iloc[:50])
probs = pipeline.predict_proba(X.iloc[50:60])
print("Quantum probabilities on 10 holdout samples:")
print(probs[:, 1])
print("Quantum test successful!")
