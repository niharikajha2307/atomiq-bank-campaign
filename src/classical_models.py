import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from src.preprocessing import build_classical_preprocessor

def get_classical_models(random_state=42):
    """
    Returns a dictionary of classical model pipelines with preprocessing included.
    All models output continuous probabilities via predict_proba.
    """
    models = {}
    
    # 1. Balanced Logistic Regression
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
    models['LogisticRegression_Balanced'] = Pipeline([
        ('preprocessor', build_classical_preprocessor()),
        ('classifier', lr)
    ])
    
    # 2. Calibrated Logistic Regression (Sigmoid / Platt calibration inside CV)
    lr_cal = CalibratedClassifierCV(
        estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
        cv=3,
        method='sigmoid'
    )
    models['LogisticRegression_Calibrated'] = Pipeline([
        ('preprocessor', build_classical_preprocessor()),
        ('classifier', lr_cal)
    ])
    
    # 3. HistGradientBoosting (handles tabular data and non-linear interactions)
    hgb = HistGradientBoostingClassifier(
        random_state=random_state,
        class_weight='balanced',
        max_iter=100,
        learning_rate=0.05,
        min_samples_leaf=10
    )
    models['HistGradientBoosting_Balanced'] = Pipeline([
        ('preprocessor', build_classical_preprocessor()),
        ('classifier', hgb)
    ])
    
    # 4. Calibrated HistGradientBoosting
    hgb_cal = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(
            random_state=random_state,
            class_weight='balanced',
            max_iter=100,
            learning_rate=0.05,
            min_samples_leaf=10
        ),
        cv=3,
        method='sigmoid'
    )
    models['HistGradientBoosting_Calibrated'] = Pipeline([
        ('preprocessor', build_classical_preprocessor()),
        ('classifier', hgb_cal)
    ])
    
    return models
