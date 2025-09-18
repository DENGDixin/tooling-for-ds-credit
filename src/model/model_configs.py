"""
Default model configurations based on hyperparameter optimization results.
These parameters were obtained from Optuna optimization runs on the German Credit Dataset.
"""

# Best parameters from hyperparameter optimization
BEST_MODEL_PARAMS = {
    'svc': {
        'C': 5.318921745689721,
        'gamma': 0.022417465302871,
        'probability': True  # Required for ensemble voting
    },
    
    'xgb': {
        'n_estimators': 200,
        'learning_rate': 0.018180006596863484,
        'max_depth': 5,
        'min_child_weight': 4,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'gamma': 4.203625511896169,
        'reg_alpha': 0.14222802126672113,
        'reg_lambda': 0.0006309962727295744,
        'scale_pos_weight': 0.33485672383934056,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
}

# Performance metrics for reference (cross-validation scores)
MODEL_PERFORMANCE = {
    'svc': {
        'cv_score': 0.6143038841053737,
        'mean_accuracy': 0.8114496245306633,
        'mean_balanced_accuracy': 0.7478482264011104,
        'mean_specificity': 0.9378017159601155,
        'mean_npv': 0.8098385060589424
    },
    
    'xgb': {
        'cv_score': 0.5820101231706246,
        'mean_accuracy': 0.7696918022528161,
        'mean_balanced_accuracy': 0.6611148710995821,
        'mean_specificity': 0.9853876369360062,
        'mean_npv': 0.7489149852280718
    }
}

# Optimization details for reference
OPTIMIZATION_INFO = {
    'svc': {
        'trials': 100,
        'optimization_metric': 'cv_ac with alpha=0.5'
    },
    'xgb': {
        'trials': 100,
        'optimization_metric': 'cv_ac with alpha=0.4'
    }
}