"""
Feature Importance Components for Streamlit App
==============================================

This module contains the feature importance analysis functionality
for the German Credit Risk Analysis Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path
sys.path.append('src')


@st.cache_data
def load_feature_importance_data():
    """Load data and models for feature importance analysis"""
    try:
        from src.utils.data_loader import load_german_credit_data, prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        from src.model.best_models import create_pipxgb_for_analysis
        from src.utils.feature_importance import extract_xgb_feature_importance, comprehensive_feature_analysis
        
        # Load and prepare data
        trainset, testset = load_german_credit_data()
        trainset_transformed, testset_transformed = apply_feature_transformations(trainset, testset)
        X, y, test_X = prepare_features_and_target(trainset_transformed, testset_transformed)
        
        # Create XGBoost pipeline for analysis
        pipxgb = create_pipxgb_for_analysis(X)
        pipxgb.fit(X, y)
        
        return {
            'X': X,
            'y': y,
            'pipxgb': pipxgb,
            'trainset': trainset_transformed
        }
    except Exception as e:
        st.error(f"Error loading feature importance data: {str(e)}")
        return None


def show_feature_importance_page():
    """Complete feature importance analysis page"""
    st.markdown('<h2 class="section-header">🔍 Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    # Load data
    data = load_feature_importance_data()
    if data is None:
        st.error("Unable to load data for feature importance analysis.")
        return
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Analysis Settings")
    
    importance_type = st.sidebar.selectbox(
        "Feature Importance Type:",
        ['gain', 'weight', 'cover'],
        index=0,
        help="Gain: Average improvement in loss, Weight: Number of times used, Cover: Number of samples affected"
    )
    
    top_n = st.sidebar.slider(
        "Number of Top Features:",
        min_value=5,
        max_value=30,
        value=15,
        help="Select how many top features to display"
    )
    
    # Main content tabs
    tab1, tab2 = st.tabs([
        "🏆 Top Features",
        "📊 Feature Categories"
    ])
    
    with tab1:
        show_top_features(data, importance_type, top_n)
    
    with tab2:
        show_feature_categories(data, importance_type)


def show_top_features(data, importance_type, top_n):
    """Show top important features"""
    st.markdown(f"### 🏆 Top {top_n} Most Important Features")
    
    try:
        from src.utils.feature_importance import extract_xgb_feature_importance
        
        # Extract feature importance
        importance_df = extract_xgb_feature_importance(
            data['pipxgb'], 
            data['X'], 
            importance_type=importance_type
        )
        
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Display as metrics
        st.markdown("#### 📊 Feature Importance Scores")
        
        # Create columns for top 5 features
        cols = st.columns(5)
        for i, (_, row) in enumerate(top_features.head(5).iterrows()):
            with cols[i]:
                st.metric(
                    f"#{i+1} {row['Feature'][:15]}...", 
                    f"{row['Importance']:.3f}",
                    help=f"Full name: {row['Feature']}"
                )
        
        # Bar chart
        fig = plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel(f'Feature Importance ({importance_type.title()})')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show detailed table
        st.markdown("#### 📋 Detailed Feature Rankings")
        st.dataframe(
            top_features.style.format({'Importance': '{:.4f}'}).bar(
                subset=['Importance'], 
                color='lightblue'
            ),
            width='stretch'
        )
        
        # Feature insights
        show_feature_insights(top_features)
        
    except Exception as e:
        st.error(f"Error analyzing top features: {str(e)}")


def show_feature_categories(data, importance_type):
    """Show feature importance by categories"""
    st.markdown("### 🎯 Feature Importance by Categories")
    
    try:
        from src.utils.feature_importance import extract_xgb_feature_importance
        
        # Extract feature importance
        importance_df = extract_xgb_feature_importance(
            data['pipxgb'], 
            data['X'], 
            importance_type=importance_type
        )
        
        # Categorize features
        categories = categorize_features(importance_df)
        
        # Show category-wise analysis
        for category, features in categories.items():
            if not features.empty:
                st.markdown(f"#### 📂 {category}")
                
                # Top 3 features in this category
                top_3 = features.head(3)
                cols = st.columns(3)
                
                for i, (_, row) in enumerate(top_3.iterrows()):
                    with cols[i]:
                        st.metric(
                            f"#{i+1} {row['Feature'][:20]}...",
                            f"{row['Importance']:.3f}",
                            help=f"Full name: {row['Feature']}"
                        )
                
                # Category chart
                if len(features) > 3:
                    fig = plt.figure(figsize=(10, 6))
                    plt.barh(range(len(features)), features['Importance'])
                    plt.yticks(range(len(features)), features['Feature'])
                    plt.xlabel(f'Feature Importance ({importance_type.title()})')
                    plt.title(f'{category} Features')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.markdown("---")
        
    except Exception as e:
        st.error(f"Error analyzing feature categories: {str(e)}")


def categorize_features(importance_df):
    """Categorize features into logical groups"""
    categories = {
        'Financial Features': importance_df[importance_df['Feature'].str.contains(
            'LoanAmount|LoanDuration|InstallmentPercent|monthly_Payment|amount_installment', 
            case=False, na=False
        )],
        'Personal Information': importance_df[importance_df['Feature'].str.contains(
            'Age|Sex|ForeignWorker|PersonalStatus', 
            case=False, na=False
        )],
        'Employment & Housing': importance_df[importance_df['Feature'].str.contains(
            'Employment|Job|Housing|ResidenceDuration', 
            case=False, na=False
        )],
        'Credit History': importance_df[importance_df['Feature'].str.contains(
            'CreditHistory|ExistingCredits|Purpose', 
            case=False, na=False
        )],
        'Engineered Features': importance_df[importance_df['Feature'].str.contains(
            'jobtime_foreign|ForeignWorker_amount_Scaled', 
            case=False, na=False
        )]
    }
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if not v.empty}
    
    return categories


def show_feature_insights(top_features):
    """Show insights about top features"""
    st.markdown("#### 💡 Feature Insights")
    
    insights = []
    
    for _, row in top_features.head(5).iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        # Generate insights based on feature names
        if 'LoanAmount' in feature:
            insights.append(f"💰 **{feature}**: Loan amount is a critical factor in credit decisions (importance: {importance:.3f})")
        elif 'LoanDuration' in feature:
            insights.append(f"⏰ **{feature}**: Loan duration significantly impacts risk assessment (importance: {importance:.3f})")
        elif 'Age' in feature:
            insights.append(f"👤 **{feature}**: Customer age is an important demographic factor (importance: {importance:.3f})")
        elif 'CreditHistory' in feature:
            insights.append(f"📊 **{feature}**: Past credit behavior is highly predictive (importance: {importance:.3f})")
        elif 'monthly_Payment' in feature:
            insights.append(f"💳 **{feature}**: Monthly payment capacity is crucial for assessment (importance: {importance:.3f})")
        else:
            insights.append(f"🔍 **{feature}**: This feature shows significant importance in the model (importance: {importance:.3f})")
    
    for insight in insights:
        st.markdown(insight)