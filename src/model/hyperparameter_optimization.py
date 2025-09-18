"""
Hyperparameter optimization utilities for German Credit Dataset
"""

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
import optuna

from src.utils.feature_engineering import FeatureAdder
from src.tools import cv_ac
from src.model.model_configs import BEST_MODEL_PARAMS
from src.model.best_models import create_preprocessor


def custom_metric(y_true, y_pred):
    """
    Custom metric combining accuracy and NPV.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Combined score
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate NPV (Negative Predictive Value)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Combine accuracy and NPV
    combined_score = 0.5 * accuracy + 0.5 * npv
    return combined_score


def create_objective_svc(X, y):
    """
    Create objective function for SVC hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        function: Objective function for Optuna
    """
    def objective_svc(trial):
        # Hyperparameter search space
        C = trial.suggest_float('C', 1e-4, 20, log=True)
        gamma = trial.suggest_float('gamma', 1e-3, 1e1, log=True)

        # Create preprocessor for this trial
        preprocessor = create_preprocessor(X)
        
        # Build the model pipeline
        model = make_pipeline(
            FeatureAdder(add_amount_foreign=True),
            preprocessor,
            SVC(C=C, gamma=gamma)
        )

        # Evaluate using cross-validation
        scores = cv_ac(model, X, y, alpha=0.5)
        return np.mean(scores)
    
    return objective_svc


def create_objective_xgb(X, y):
    """
    Create objective function for XGBoost hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        function: Objective function for Optuna
    """
    def objective_xgb(trial):
        # Hyperparameter search space
        param = {
            # High Impact
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),

            # Important for Overfitting and Generalization
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),

            # Secondary Importance for Regularization and Loss Reduction
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),

            # Imbalanced Dataset Adjustment (Optional)
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0, log=True),

            # XGBoost Settings
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

        # Create preprocessor for this trial
        preprocessor = create_preprocessor(X)

        # Build the model pipeline
        model = make_pipeline(
            FeatureAdder(add_amount_foreign=True),
            preprocessor,
            xgb.XGBClassifier(**param)
        )

        # Evaluate using cross-validation
        scores = cv_ac(model, X, y, alpha=0.4)
        return np.mean(scores)
    
    return objective_xgb


def create_objective_rf(X, y):
    """
    Create objective function for RandomForest hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        function: Objective function for Optuna
    """
    def objective_rf(trial):
        # Hyperparameter search space
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
        }

        # Create preprocessor for this trial
        preprocessor = create_preprocessor(X)

        # Build the model pipeline
        model = make_pipeline(
            FeatureAdder(add_amount_foreign=True),
            preprocessor,
            RandomForestClassifier(**param, random_state=42)
        )

        # Evaluate using cross-validation
        scores = cv_ac(model, X, y, alpha=0.4)
        return np.mean(scores)
    
    return objective_rf


def create_objective_histgbc(X, y):
    """
    Create objective function for HistGradientBoosting hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        function: Objective function for Optuna
    """
    def objective_histgbc(trial):
        # Hyperparameter search space
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100, step=10),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100, step=10),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 1.0, log=True),
            'early_stopping': True,
            'random_state': 42
        }

        # Create preprocessor for this trial
        preprocessor = create_preprocessor(X)

        # Build the model pipeline
        model = make_pipeline(
            FeatureAdder(add_amount_foreign=True),
            preprocessor,
            HistGradientBoostingClassifier(**param)
        )

        # Evaluate using cross-validation
        scores = cv_ac(model, X, y, alpha=0.4)
        return np.mean(scores)
    
    return objective_histgbc


def optimize_all_models(X, y, n_trials=100):
    """
    Optimize hyperparameters for all models.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of optimization trials
        
    Returns:
        dict: Best parameters for each model
    """
    results = {}
    
    # Optimize SVC
    print("Optimizing SVC...")
    study_svc = optuna.create_study(direction='maximize', study_name='SVC Hyperparameter Optimization')
    study_svc.optimize(create_objective_svc(X, y), n_trials=n_trials)
    results['svc'] = study_svc.best_params
    
    # Optimize XGBoost
    print("Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction='maximize', study_name='XGB Hyperparameter Optimization')
    study_xgb.optimize(create_objective_xgb(X, y), n_trials=n_trials)
    results['xgb'] = study_xgb.best_params
    
    # Optimize RandomForest
    print("Optimizing RandomForest...")
    study_rf = optuna.create_study(direction='maximize', study_name='RF Hyperparameter Optimization')
    study_rf.optimize(create_objective_rf(X, y), n_trials=20)  # Fewer trials for RF
    results['rf'] = study_rf.best_params
    
    # Optimize HistGradientBoosting
    print("Optimizing HistGradientBoosting...")
    study_histgbc = optuna.create_study(direction='maximize', study_name='HistGBC Hyperparameter Optimization')
    study_histgbc.optimize(create_objective_histgbc(X, y), n_trials=n_trials)
    results['histgbc'] = study_histgbc.best_params
    
    return results


# Convenience functions for easy single-model optimization

def optimize_svc_quick(X, y, n_trials=50):
    """
    Quick SVC optimization with fewer trials.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of trials (default: 50)
    
    Returns:
        dict: Best parameters and score
    """
    print("🚀 Quick SVC Optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(create_objective_svc(X, y), n_trials=n_trials)
    
    print(f"✅ Best SVC Score: {study.best_value:.6f}")
    print(f"📊 Best Parameters: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


def optimize_xgb_quick(X, y, n_trials=50):
    """
    Quick XGBoost optimization with fewer trials.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of trials (default: 50)
    
    Returns:
        dict: Best parameters and score
    """
    print("🚀 Quick XGBoost Optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(create_objective_xgb(X, y), n_trials=n_trials)
    
    print(f"✅ Best XGBoost Score: {study.best_value:.6f}")
    print(f"📊 Best Parameters: {study.best_params}")
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }


def run_quick_optimization_demo(X, y):
    """
    Run a quick demonstration of hyperparameter optimization for all models.
    Uses fewer trials (10-20) for fast results.
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        dict: Results for all models
    """
    print("🎯 QUICK HYPERPARAMETER OPTIMIZATION DEMO")
    print("=" * 50)
    print("⚡ Using fewer trials for demonstration purposes")
    print("💡 For production use, increase n_trials to 100+ per model\n")
    
    results = {}
    
    # Quick SVC optimization
    svc_result = optimize_svc_quick(X, y, n_trials=10)
    results['svc'] = svc_result
    print()
    
    # Quick XGBoost optimization  
    xgb_result = optimize_xgb_quick(X, y, n_trials=10)
    results['xgb'] = xgb_result
    print()
    
    print("🏁 Quick optimization demo completed!")
    print("📈 To run full optimization with more trials, use optimize_all_models(X, y, n_trials=100)")
    
    return results