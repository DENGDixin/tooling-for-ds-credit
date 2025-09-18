"""
German Credit Risk Analysis - Streamlit App
===========================================

A comprehensive web application for German credit risk analysis using machine learning.
Replaces the Jupyter notebook with an interactive web interface.

Features:
- Data exploration and visualization
- Model evaluation and comparison
- Feature importance analysis
- Hyperparameter optimization
- Real-time predictions
- Interactive visualizations

Author: Data Science Team
Date: September 2025
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="German Credit Risk Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 German Credit Risk Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "🏠 Home",
            "📈 Data Exploration", 
            "🧪 Model Evaluation",
            "🔍 Feature Importance",
            "⚙️ Hyperparameter Optimization",
            "🎯 Make Predictions"
        ]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Data Exploration":
        show_data_exploration_page()
    elif page == "🧪 Model Evaluation":
        show_model_evaluation_page()
    elif page == "🔍 Feature Importance":
        show_feature_importance_page()
    elif page == "⚙️ Hyperparameter Optimization":
        show_hyperparameter_optimization_page()
    elif page == "🎯 Make Predictions":
        show_prediction_page()


def show_home_page():
    """Display the home page with overview and quick stats"""
    
    st.markdown('<h2 class="section-header">Welcome to German Credit Risk Analysis</h2>', unsafe_allow_html=True)
    
    # Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a Streamlit web application for the Kaggle competition: [DSB-24 German Credit](https://www.kaggle.com/competitions/dsb-24-german-credit).
        
        The application uses a **custom business-oriented metric** that considers the financial impact of loan decisions, taking into account loan amounts and real-world costs of false positives and false negatives in credit risk assessment.
        
        **🏆 Model Performance Summary:**
        - **🥇 Best Model**: SVC achieves **-103.5902** on custom business metric
        - **📊 Custom Metric**: Cost-sensitive scoring considering loan amounts and real-world impact
        - **🔍 Explainable AI**: XGBoost excels at feature importance for transparent decisions
        
        **Key Features:**
        - **Data Exploration**: Interactive visualizations and statistical analysis
        - **Model Comparison**: Performance evaluation using business-oriented metrics
        - **Feature Analysis**: Understanding which factors drive credit decisions with XGBoost
        - **Hyperparameter Tuning**: Optimize model performance with custom parameters
        - **Real-time Predictions**: Make instant credit risk assessments
        
        ### 🤖 Model Insights
        
        **🥇 Support Vector Classifier (SVC) - Champion Model**
        - **Custom Metric Score**: -103.5902 (Best performance)
        - **Strength**: Superior cost-sensitive classification
        - **Use Case**: Production deployment for real-world credit decisions
        
        **🔍 XGBoost Classifier - Explainability Champion**
        - **Custom Metric Score**: -96.7352 (Strong performance)
        - **Strength**: Outstanding feature importance analysis and model interpretability
        - **Use Case**: Understanding credit risk factors and regulatory compliance
        """)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **📊 Quick Stats**
        
        - **Dataset**: German Credit Data
        - **Records**: 1,000 customers
        - **Features**: 20+ variables
        - **Target**: Credit risk (Good/Bad)
        
        **🏆 Champion Model**
        - **Best**: SVC (-103.5902 custom metric)
        - **Optimization**: Optuna-based tuning
        - **Business Focus**: Cost-sensitive evaluation
        
        **🔍 Explainable Model**
        - **XGBoost**: Premier choice for feature importance
        - **Transparency**: Clear factor attribution
        - **Regulatory**: Compliance-ready explanations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System status
    st.markdown('<h3 class="section-header">🔧 System Status</h3>', unsafe_allow_html=True)
    
    try:
        # Test imports
        from src.utils.data_loader import load_german_credit_data
        from src.model.model_configs import BEST_MODEL_PARAMS, MODEL_PERFORMANCE
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("✅ **System Ready**: All modules loaded successfully")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show model performance summary
        st.markdown("### 🏆 Model Performance Comparison")
        
        # Create performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🥇 SVC - Production Champion**
            - Custom Business Metric: **-103.5902** ⭐
            - Best for: Real-world deployment
            - Strength: Cost-sensitive optimization
            """)
            st.success("✅ Recommended for production use")
        
        with col2:
            st.markdown("""
            **🔍 XGBoost - Explainability Expert**
            - Custom Business Metric: **-96.7352**
            - Best for: Feature importance analysis
            - Strength: Model interpretability & transparency
            """)
            st.info("🔍 Ideal for explainable ML requirements")
        
        # Performance metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        # Custom metric explanation
        st.markdown("---")
        st.markdown("### 📈 About the Custom Business Metric")
        
        with st.expander("🔍 Click to understand the cost-sensitive evaluation"):
            st.markdown("""  
            **The Custom Metric Formula:**
            ```
            def compute_costs(LoanAmount):
                return({'Risk_No Risk': 5.0 + .6 * LoanAmount, 'No Risk_No Risk': 1.0 - .05 * LoanAmount,
                    'Risk_Risk': 1.0, 'No Risk_Risk': 1.0})

            def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
                '''
                A custom metric for the German credit dataset
                '''
                real_prop = {'Risk': .02, 'No Risk': .98}
                train_prop = {'Risk': 1/3, 'No Risk': 2/3}
                custom_weight = {'Risk': real_prop['Risk']/train_prop['Risk'], 'No Risk': real_prop['No Risk']/train_prop['No Risk']}
                costs = compute_costs(solution['LoanAmount'])
                y_true = solution['Risk']
                y_pred = submission['Risk']
                loss = (y_true=='Risk') * custom_weight['Risk'] * ((y_pred=='Risk') * costs['Risk_Risk'] + (y_pred=='No Risk') * costs['Risk_No Risk']) + (y_true=='No Risk') * custom_weight['No Risk'] * ((y_pred=='Risk') * costs['No Risk_Risk'] + (y_pred=='No Risk') * costs['No Risk_No Risk'])
                return loss.mean()

            ```
            """)
        
    except Exception as e:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"⚠️ **Setup Required**: {str(e)}")
        st.markdown("Please ensure all dependencies are installed and the src/ directory is properly set up.")
        st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the German credit data"""
    try:
        from src.utils.data_loader import load_german_credit_data, prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        
        # Load datasets
        trainset, testset = load_german_credit_data()
        
        # Apply transformations
        trainset_transformed, testset_transformed = apply_feature_transformations(trainset, testset)
        
        # Prepare features and target
        X, y, test_X = prepare_features_and_target(trainset_transformed, testset_transformed)
        
        return {
            'trainset': trainset,
            'testset': testset,
            'trainset_transformed': trainset_transformed,
            'testset_transformed': testset_transformed,
            'X': X,
            'y': y,
            'test_X': test_X
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def show_data_exploration_page():
    """Data exploration and visualization page"""
    st.markdown('<h2 class="section-header">📈 Data Exploration</h2>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Unable to load data. Please check your setup.")
        return
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", len(data['trainset']))
    with col2:
        st.metric("Test Samples", len(data['testset']))
    with col3:
        st.metric("Features", data['X'].shape[1])
    with col4:
        good_ratio = (data['y'] == 1).mean()
        st.metric("Good Credit %", f"{good_ratio:.1%}")
    
    # Data preview
    st.markdown("### 🔍 Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Training Data", "Test Data", "Feature Info"])
    
    with tab1:
        st.dataframe(data['trainset'].head(10), width='stretch')
        
    with tab2:
        st.dataframe(data['testset'].head(10), width='stretch')
    
    with tab3:
        # Fix Arrow serialization error by converting dtypes to string
        dtype_df = data['X'].dtypes.to_frame('Data Type')
        dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
        st.dataframe(dtype_df, width='stretch')    # Visualizations
    st.markdown("### 📊 Data Visualizations")
    
    viz_type = st.selectbox(
        "Choose visualization:",
        ["Target Distribution", "Feature Distributions", "Correlation Matrix"]
    )
    
    if viz_type == "Target Distribution":
        show_target_distribution(data['y'])
    elif viz_type == "Feature Distributions":
        show_feature_distributions(data['trainset'])
    elif viz_type == "Correlation Matrix":
        show_correlation_matrix(data['X'])


def show_target_distribution(y):
    """Show target variable distribution"""
    fig = px.pie(
        values=y.value_counts().values,
        names=['Bad Credit', 'Good Credit'],
        title="Credit Risk Distribution",
        color_discrete_map={'Good Credit': '#2E8B57', 'Bad Credit': '#DC143C'}
    )
    st.plotly_chart(fig, width='stretch')


def show_feature_distributions(df):
    """Show feature distributions"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        selected_feature = st.selectbox("Select feature:", numeric_cols)
        
        fig = px.histogram(
            df, 
            x=selected_feature,
            title=f"Distribution of {selected_feature}",
            nbins=30
        )
        st.plotly_chart(fig, width='stretch')


def show_correlation_matrix(X):
    """Show correlation matrix"""
    # Select numeric columns only
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    corr_matrix = X[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, width='stretch')


def show_model_evaluation_page():
    """Model evaluation and comparison page"""
    try:
        from streamlit_components.model_evaluation import show_model_evaluation_page as show_eval
        show_eval()
    except ImportError:
        st.error("Model evaluation component not found. Please check the streamlit_components directory.")


def show_feature_importance_page():
    """Feature importance analysis page"""
    try:
        from streamlit_components.feature_importance import show_feature_importance_page as show_feat
        show_feat()
    except ImportError:
        st.error("Feature importance component not found. Please check the streamlit_components directory.")


def show_hyperparameter_optimization_page():
    """Hyperparameter optimization page"""
    try:
        from streamlit_components.hyperparameter_optimization import show_hyperparameter_optimization_page as show_hyper
        show_hyper()
    except ImportError:
        st.error("Hyperparameter optimization component not found. Please check the streamlit_components directory.")


def show_prediction_page():
    """Prediction page for new data"""
    try:
        from streamlit_components.prediction import show_prediction_page as show_pred
        show_pred()
    except ImportError:
        st.error("Prediction component not found. Please check the streamlit_components directory.")


if __name__ == "__main__":
    main()