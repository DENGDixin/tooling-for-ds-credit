"""
Utility functions for accessing optimized model configurations and performance metrics.
"""

from src.model.model_configs import BEST_MODEL_PARAMS, MODEL_PERFORMANCE, OPTIMIZATION_INFO


def get_best_params(model_name=None):
    """
    Get the best hyperparameters for a specific model or all models.
    
    Args:
        model_name (str, optional): Name of the model ('xgb', 'svc', 'rf', 'histgbc').
                                   If None, returns all parameters.
    
    Returns:
        dict: Best hyperparameters for the specified model or all models
    """
    if model_name is None:
        return BEST_MODEL_PARAMS
    
    if model_name not in BEST_MODEL_PARAMS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(BEST_MODEL_PARAMS.keys())}")
    
    return BEST_MODEL_PARAMS[model_name]


def get_model_performance(model_name=None):
    """
    Get the performance metrics for a specific model or all models.
    
    Args:
        model_name (str, optional): Name of the model ('xgb', 'svc', 'rf', 'histgbc').
                                   If None, returns all performance metrics.
    
    Returns:
        dict: Performance metrics for the specified model or all models
    """
    if model_name is None:
        return MODEL_PERFORMANCE
    
    if model_name not in MODEL_PERFORMANCE:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_PERFORMANCE.keys())}")
    
    return MODEL_PERFORMANCE[model_name]


def get_optimization_info(model_name=None):
    """
    Get the optimization information for a specific model or all models.
    
    Args:
        model_name (str, optional): Name of the model ('xgb', 'svc', 'rf', 'histgbc').
                                   If None, returns all optimization info.
    
    Returns:
        dict: Optimization details for the specified model or all models
    """
    if model_name is None:
        return OPTIMIZATION_INFO
    
    if model_name not in OPTIMIZATION_INFO:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(OPTIMIZATION_INFO.keys())}")
    
    return OPTIMIZATION_INFO[model_name]


def print_model_summary(model_name):
    """
    Print a comprehensive summary of a model's configuration and performance.
    
    Args:
        model_name (str): Name of the model ('xgb', 'svc', 'rf', 'histgbc')
    """
    if model_name not in BEST_MODEL_PARAMS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(BEST_MODEL_PARAMS.keys())}")
    
    print(f"\n=== {model_name.upper()} Model Summary ===")
    
    # Optimization info
    opt_info = get_optimization_info(model_name)
    print(f"Optimization: {opt_info['trials']} trials using {opt_info['optimization_metric']}")
    
    # Performance metrics
    perf = get_model_performance(model_name)
    print(f"Cross-validation Score: {perf['cv_score']:.6f}")
    
    if 'mean_accuracy' in perf:
        print(f"Mean Accuracy: {perf['mean_accuracy']:.6f}")
        print(f"Mean Balanced Accuracy: {perf['mean_balanced_accuracy']:.6f}")
        print(f"Mean Specificity: {perf['mean_specificity']:.6f}")
        print(f"Mean NPV: {perf['mean_npv']:.6f}")
    
    # Best parameters
    params = get_best_params(model_name)
    print("\nBest Parameters:")
    for param, value in params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")


def print_all_models_summary():
    """
    Print a summary of all models' performance and configurations.
    """
    print("=" * 60)
    print("GERMAN CREDIT DATASET - OPTIMIZED MODEL CONFIGURATIONS")
    print("=" * 60)
    
    models = ['svc', 'xgb', 'rf', 'histgbc']
    
    print("\nModel Performance Ranking (by CV Score):")
    performances = [(name, get_model_performance(name)['cv_score']) for name in models]
    performances.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(performances, 1):
        print(f"{i}. {name.upper()}: {score:.6f}")
    
    for model in models:
        print_model_summary(model)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    print_all_models_summary()