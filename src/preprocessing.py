import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

NUMERIC_COLS = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CATEGORICAL_COLS = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

class PdaysFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms the 'pdays' column into two clean signals:
    1. 'pdays_contacted': binary indicator (1 if pdays != -1, else 0)
    2. 'pdays_scaled': continuous value with -1 mapped to 0 (or baseline), so -1 doesn't distort numerical scaling.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if 'pdays' in X_out.columns:
            pdays_numeric = pd.to_numeric(X_out['pdays'], errors='coerce').fillna(-1)
            X_out['pdays_contacted'] = (pdays_numeric != -1).astype(float)
            # When customer was not previously contacted (-1), set days to 0 for numerical scaling
            X_out['pdays'] = np.where(pdays_numeric == -1, 0.0, pdays_numeric)
        
        # Ensure all numeric columns are strictly float
        for col in NUMERIC_COLS:
            if col in X_out.columns:
                X_out[col] = pd.to_numeric(X_out[col], errors='coerce').fillna(0.0)
        
        # Ensure all categorical columns are strings
        for col in CATEGORICAL_COLS:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype(str)
                
        return X_out

def build_classical_preprocessor():
    """
    Builds a ColumnTransformer for classical models:
    - Numeric features: RobustScaler (resilient to outliers in balance/campaign)
    - Categorical features: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    extended_numeric = NUMERIC_COLS + ['pdays_contacted']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), extended_numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('engineer', PdaysFeatureEngineer()),
        ('preprocess', preprocessor)
    ])
    return pipeline

def build_quantum_preprocessor(k_features=5):
    """
    Builds a ColumnTransformer for quantum models:
    - Preprocessing identical to classical
    - SelectKBest(f_classif, k=k_features) fitted strictly on training data
    - MinMaxScaler(feature_range=(0, np.pi)): scales features into [0, pi] for quantum angle rotation
    """
    extended_numeric = NUMERIC_COLS + ['pdays_contacted']
    
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), extended_numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('engineer', PdaysFeatureEngineer()),
        ('preprocess', base_preprocessor),
        ('selector', SelectKBest(score_func=f_classif, k=k_features)),
        ('scaler_quantum', MinMaxScaler(feature_range=(0, np.pi)))
    ])
    return pipeline
