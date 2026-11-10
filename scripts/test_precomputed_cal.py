import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

K = np.eye(20)
y = np.array([0]*10 + [1]*10)

svc = SVC(kernel='precomputed', random_state=42)
cal = CalibratedClassifierCV(svc, cv=3)
try:
    cal.fit(K, y)
    probs = cal.predict_proba(K[:5])
    print("CalibratedClassifierCV on precomputed kernel works!")
except Exception as e:
    print("Error with CalibratedClassifierCV:", e)

# Test SVC with probability=True
svc_prob = SVC(kernel='precomputed', probability=True, random_state=42)
svc_prob.fit(K, y)
probs2 = svc_prob.predict_proba(K[:5])
print("SVC(probability=True) works!")
