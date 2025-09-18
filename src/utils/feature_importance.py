"""
Feature importance analysis utilities for German Credit Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance


def extract_xgb_feature_importance(pipeline, X_data=None, importance_type='gain'):
    """
    Extract feature importance from XGBoost model in a pipeline.
    
    Args:
        pipeline: sklearn pipeline containing XGBoost model
        X_data: pd.DataFrame, input data to get feature names (optional if pipeline is fitted)
        importance_type: str, type of importance ('gain', 'weight', 'cover')
        
    Returns:
        pd.DataFrame: Feature importance dataframe sorted by importance
    """
    # Get the XGBoost model from pipeline
    model = pipeline.named_steps['xgbclassifier']
    
    # Get the booster object
    booster = model.get_booster()
    
    # Get feature importance
    importance = booster.get_score(importance_type=importance_type)
    
    # Get feature names from the pipeline if possible
    try:
        # Try to get feature names from the fitted pipeline
        feature_names = pipeline.named_steps['columntransformer'].get_feature_names_out()
    except:
        # If that fails and X_data is provided, create a dummy transformation to get names
        if X_data is not None:
            # Create a copy of the pipeline and fit just the preprocessing steps
            from sklearn.base import clone
            temp_pipeline = clone(pipeline)
            # Transform data through preprocessing steps to get feature names
            X_mid = temp_pipeline.named_steps['featureadder'].fit_transform(X_data)
            X_transformed = temp_pipeline.named_steps['columntransformer'].fit_transform(X_mid)
            feature_names = temp_pipeline.named_steps['columntransformer'].get_feature_names_out()
        else:
            # Fallback: use generic feature names
            feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    # Map feature importance to actual feature names
    importance_named = {}
    for k, v in importance.items():
        try:
            # XGBoost uses f0, f1, etc. format
            feature_idx = int(k[1:])
            if feature_idx < len(feature_names):
                importance_named[feature_names[feature_idx]] = v
        except:
            # If parsing fails, use the original key
            importance_named[k] = v
    
    # Create DataFrame and sort by importance
    imp_df = pd.DataFrame(importance_named.items(), columns=['Feature', 'Importance'])
    imp_df = imp_df.sort_values(by='Importance', ascending=False, ignore_index=True)
    
    return imp_df


def plot_feature_importance(imp_df, top_n=20, figsize=(10, 8), title_suffix=""):
    """
    Plot feature importance as a horizontal bar chart.
    
    Args:
        imp_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: int, number of top features to display
        figsize: tuple, figure size
        title_suffix: str, additional text for plot title
    """
    # Select top N features
    top_features = imp_df.head(top_n)
    
    # Create horizontal bar plot
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Importance{title_suffix}')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, v in enumerate(top_features['Importance']):
        plt.text(v + max(top_features['Importance']) * 0.01, i, f'{v:.2f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def analyze_feature_importance_by_type(imp_df, feature_prefixes=None):
    """
    Analyze feature importance grouped by feature type/prefix.
    
    Args:
        imp_df: DataFrame with 'Feature' and 'Importance' columns
        feature_prefixes: dict, mapping of prefix to category name
                         If None, uses default German credit prefixes
        
    Returns:
        pd.DataFrame: Summary of importance by feature type
    """
    if feature_prefixes is None:
        feature_prefixes = {
            'onehot__': 'Categorical Features',
            'scaler__': 'Numerical Features',
            'scaler__monthly_Payment': 'Engineered: Monthly Payment',
            'scaler__amount_installment': 'Engineered: Amount Installment',
            'scaler__ForeignWorker_amount_Scaled': 'Engineered: Foreign Worker Amount'
        }
    
    # Create feature type column
    imp_df_copy = imp_df.copy()
    imp_df_copy['Feature_Type'] = 'Other'
    
    for prefix, category in feature_prefixes.items():
        mask = imp_df_copy['Feature'].str.startswith(prefix)
        imp_df_copy.loc[mask, 'Feature_Type'] = category
    
    # Group by feature type and calculate statistics
    type_summary = imp_df_copy.groupby('Feature_Type').agg({
        'Importance': ['count', 'sum', 'mean', 'std', 'max']
    }).round(4)
    
    # Flatten column names
    type_summary.columns = ['Feature_Count', 'Total_Importance', 'Mean_Importance', 
                           'Std_Importance', 'Max_Importance']
    
    # Calculate percentage of total importance
    total_importance = imp_df_copy['Importance'].sum()
    type_summary['Importance_Percentage'] = (
        type_summary['Total_Importance'] / total_importance * 100
    ).round(2)
    
    # Sort by total importance
    type_summary = type_summary.sort_values('Total_Importance', ascending=False)
    
    return type_summary


def plot_importance_by_type(type_summary, figsize=(10, 6)):
    """
    Plot feature importance summary by feature type.
    
    Args:
        type_summary: DataFrame from analyze_feature_importance_by_type()
        figsize: tuple, figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Total importance by type
    ax1.barh(range(len(type_summary)), type_summary['Total_Importance'])
    ax1.set_yticks(range(len(type_summary)))
    ax1.set_yticklabels(type_summary.index)
    ax1.set_xlabel('Total Importance')
    ax1.set_title('Total Feature Importance by Type')
    ax1.invert_yaxis()
    
    # Add percentage labels
    for i, (total, pct) in enumerate(zip(type_summary['Total_Importance'], 
                                        type_summary['Importance_Percentage'])):
        ax1.text(total + max(type_summary['Total_Importance']) * 0.01, i, 
                f'{pct:.1f}%', va='center', fontsize=9)
    
    # Plot 2: Mean importance by type
    ax2.barh(range(len(type_summary)), type_summary['Mean_Importance'])
    ax2.set_yticks(range(len(type_summary)))
    ax2.set_yticklabels(type_summary.index)
    ax2.set_xlabel('Mean Importance')
    ax2.set_title('Average Feature Importance by Type')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def get_top_features_by_category(imp_df, n_per_category=5):
    """
    Get top N most important features from each category.
    
    Args:
        imp_df: DataFrame with 'Feature' and 'Importance' columns
        n_per_category: int, number of top features per category
        
    Returns:
        dict: Dictionary with category as key and top features as values
    """
    categories = {
        'Categorical': imp_df[imp_df['Feature'].str.startswith('onehot__')],
        'Numerical': imp_df[imp_df['Feature'].str.startswith('scaler__') & 
                           ~imp_df['Feature'].str.contains('monthly_Payment|amount_installment|ForeignWorker_amount_Scaled')],
        'Engineered': imp_df[imp_df['Feature'].str.contains('monthly_Payment|amount_installment|ForeignWorker_amount_Scaled')]
    }
    
    top_by_category = {}
    for category, features in categories.items():
        if not features.empty:
            top_features = features.head(n_per_category)
            top_by_category[category] = top_features[['Feature', 'Importance']].to_dict('records')
    
    return top_by_category


def comprehensive_feature_analysis(pipeline, X_data, importance_type='gain', 
                                  top_n=20, save_plots=False, plot_dir='plots'):
    """
    Perform comprehensive feature importance analysis.
    
    Args:
        pipeline: sklearn pipeline containing XGBoost model
        X_data: pd.DataFrame, input data for feature transformation
        importance_type: str, type of importance ('gain', 'weight', 'cover')
        top_n: int, number of top features to display in plots
        save_plots: bool, whether to save plots to files
        plot_dir: str, directory to save plots
        
    Returns:
        dict: Dictionary containing all analysis results
    """
    print(f"🔍 COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS ({importance_type.upper()})")
    print("=" * 60)
    
    # Extract feature importance
    imp_df = extract_xgb_feature_importance(pipeline, X_data, importance_type)
    
    # Basic statistics
    print(f"📊 Total Features: {len(imp_df)}")
    print(f"📈 Top Feature: {imp_df.iloc[0]['Feature']} (Importance: {imp_df.iloc[0]['Importance']:.4f})")
    print(f"📉 Lowest Feature: {imp_df.iloc[-1]['Feature']} (Importance: {imp_df.iloc[-1]['Importance']:.4f})")
    print(f"📊 Mean Importance: {imp_df['Importance'].mean():.4f}")
    print(f"📊 Std Importance: {imp_df['Importance'].std():.4f}")
    
    # Plot overall feature importance
    print(f"\n🎯 Plotting Top {top_n} Features...")
    plot_feature_importance(imp_df, top_n=top_n, title_suffix=f" ({importance_type.upper()})")
    
    # Analyze by feature type
    print("\n🏷️  Analyzing by Feature Type...")
    type_summary = analyze_feature_importance_by_type(imp_df)
    print(type_summary)
    
    # Plot by type
    plot_importance_by_type(type_summary)
    
    # Get top features by category
    top_by_category = get_top_features_by_category(imp_df)
    print("\n🥇 Top Features by Category:")
    for category, features in top_by_category.items():
        print(f"\n{category}:")
        for i, feature in enumerate(features, 1):
            print(f"  {i}. {feature['Feature']}: {feature['Importance']:.4f}")
    
    return {
        'importance_df': imp_df,
        'type_summary': type_summary,
        'top_by_category': top_by_category,
        'importance_type': importance_type
    }