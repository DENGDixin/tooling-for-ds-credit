"""
Model Evaluation Components for Streamlit App
============================================

This module contains the model evaluation functionality
for the German Credit Risk Analysis Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import sys

# Add src to path
sys.path.append('src')


@st.cache_data
def load_models():
    """Load and cache trained models"""
    try:
        from src.model.best_models import create_best_models
        from src.model.model_configs import BEST_MODEL_PARAMS, MODEL_PERFORMANCE
        from src.utils.data_loader import load_german_credit_data, prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        
        # Load and prepare data
        trainset, testset = load_german_credit_data()
        trainset_transformed, testset_transformed = apply_feature_transformations(trainset, testset)
        X, y, test_X = prepare_features_and_target(trainset_transformed, testset_transformed)
        
        # Create models (only XGB and SVC)
        best_models = create_best_models(X)
        
        # Fit models on training data
        for model_name, model in best_models.items():
            model.fit(X, y)
        
        return {
            'models': best_models,
            'X': X,
            'y': y,
            'test_X': test_X,
            'performance': MODEL_PERFORMANCE
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def show_model_evaluation_page():
    """Complete model evaluation page"""
    st.markdown('<h2 class="section-header">🧪 Model Evaluation & Comparison</h2>', unsafe_allow_html=True)
    
    # Load models
    model_data = load_models()
    if model_data is None:
        st.error("Unable to load models. Please check your setup.")
        return
    
    # Sidebar for model selection
    st.sidebar.markdown("### 🎛️ Model Selection")
    available_models = list(model_data['models'].keys())
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        available_models,
        default=['svc', 'xgb']
    )
    
    if not selected_models:
        st.warning("Please select at least one model to evaluate.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Overview", 
        "📈 Detailed Metrics", 
        "🎯 ROC Analysis", 
        "🧮 Confusion Matrix",
        "📋 Classification Report"
    ])
    
    with tab1:
        show_performance_overview(model_data, selected_models)
    
    with tab2:
        show_detailed_metrics(model_data, selected_models)
    
    with tab3:
        show_roc_analysis(model_data, selected_models)
    
    with tab4:
        show_confusion_matrix_analysis(model_data, selected_models)
    
    with tab4:
        show_confusion_matrix_analysis(model_data, selected_models)
    
    with tab5:
        show_classification_report_analysis(model_data, selected_models)


def calculate_comprehensive_metrics(model, X, y):
    """Calculate comprehensive metrics including accuracy, balanced accuracy, and custom metrics"""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    from sklearn.model_selection import cross_val_score
    from src.tools import leaderboad
    
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate basic metrics
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # ROC AUC if probabilities available
    roc_auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None
    
    # Custom leaderboard score (German Credit specific)
    leaderboard_score = leaderboad(y_pred, X, y)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Custom metrics for credit risk
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (recall)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value (precision)
    
    # Custom combined metric (weighted accuracy considering cost of false negatives)
    # In credit risk, false negatives (missing bad credit) are more costly
    cost_weighted_accuracy = (2 * tp + tn) / (2 * (tp + fn) + (tn + fp)) if (tp + fn + tn + fp) > 0 else 0
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'leaderboard_score': leaderboard_score,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'npv': npv,
        'ppv': ppv,
        'cost_weighted_accuracy': cost_weighted_accuracy,
        'cv_accuracy': cv_accuracy,
        'cv_std': cv_scores.std()
    }


def show_performance_overview(model_data, selected_models):
    """Show performance overview with comprehensive metrics"""
    st.markdown("### 🏆 Model Performance Summary")
    
    # Calculate comprehensive metrics for each model
    performance_data = {}
    
    with st.spinner("Calculating comprehensive metrics..."):
        for model_name in selected_models:
            model = model_data['models'][model_name]
            metrics = calculate_comprehensive_metrics(model, model_data['X'], model_data['y'])
            performance_data[model_name] = metrics
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(performance_data).T
    
    # Display key metrics cards
    st.markdown("#### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**🎯 Accuracy**")
        for model_name in selected_models:
            accuracy = performance_data[model_name]['accuracy']
            st.metric(
                f"{model_name.upper()}", 
                f"{accuracy:.4f}",
                help="Proportion of correct predictions"
            )
    
    with col2:
        st.markdown("**⚖️ Balanced Accuracy**")
        for model_name in selected_models:
            bal_acc = performance_data[model_name]['balanced_accuracy']
            st.metric(
                f"{model_name.upper()}", 
                f"{bal_acc:.4f}",
                help="Average of sensitivity and specificity"
            )
    
    with col3:
        st.markdown("**🏆 Leaderboard Score**")
        for model_name in selected_models:
            leaderboard = performance_data[model_name]['leaderboard_score']
            st.metric(
                f"{model_name.upper()}", 
                f"{leaderboard:.4f}",
                help="Custom German Credit competition score (lower is better)"
            )
    
    with col4:
        st.markdown("**💰 Cost-Weighted Accuracy**")
        for model_name in selected_models:
            cost_acc = performance_data[model_name]['cost_weighted_accuracy']
            st.metric(
                f"{model_name.upper()}", 
                f"{cost_acc:.4f}",
                help="Accuracy weighted for credit risk costs"
            )
    
    with col5:
        st.markdown("**📈 Cross-Val Accuracy**")
        for model_name in selected_models:
            cv_acc = performance_data[model_name]['cv_accuracy']
            cv_std = performance_data[model_name]['cv_std']
            st.metric(
                f"{model_name.upper()}", 
                f"{cv_acc:.4f}",
                delta=f"±{cv_std:.4f}",
                help="5-fold cross-validation accuracy"
            )
    
    # Comprehensive metrics table
    st.markdown("#### 📋 Detailed Metrics Comparison")
    
    display_metrics = [
        'accuracy', 'balanced_accuracy', 'leaderboard_score', 'precision', 'recall', 'f1_score', 
        'specificity', 'sensitivity', 'roc_auc', 'cost_weighted_accuracy'
    ]
    
    display_df = metrics_df[display_metrics].round(4)
    display_df.index = [name.upper() for name in display_df.index]
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Accuracy', 
        'leaderboard_score': 'Leaderboard Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'specificity': 'Specificity',
        'sensitivity': 'Sensitivity',
        'roc_auc': 'ROC AUC',
        'cost_weighted_accuracy': 'Cost-Weighted Accuracy'
    })
    
    # Style the dataframe (use reverse color map for leaderboard score since lower is better)
    styled_df = display_df.style.format("{:.4f}")
    
    # Apply different color schemes: green for metrics where higher is better, red for leaderboard score where lower is better
    for col in display_df.columns:
        if col == 'Leaderboard Score':
            styled_df = styled_df.background_gradient(cmap='RdYlGn_r', subset=[col])  # Reverse colormap
        else:
            styled_df = styled_df.background_gradient(cmap='RdYlGn', subset=[col])
    
    st.dataframe(styled_df, width='stretch')
    
    # Performance comparison chart
    st.markdown("#### 📊 Visual Performance Comparison")
    
    # Create radar chart for comparison
    categories = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Leaderboard Score (Inverted)']
    
    fig = go.Figure()
    
    for model_name in selected_models:
        # For leaderboard score, invert it (1 - normalized score) since lower is better
        # Normalize leaderboard score to 0-1 range for visualization
        lb_scores = [performance_data[m]['leaderboard_score'] for m in selected_models]
        min_lb, max_lb = min(lb_scores), max(lb_scores)
        normalized_lb = 1 - ((performance_data[model_name]['leaderboard_score'] - min_lb) / (max_lb - min_lb)) if max_lb != min_lb else 0.5
        
        values = [
            performance_data[model_name]['accuracy'],
            performance_data[model_name]['balanced_accuracy'],
            performance_data[model_name]['precision'],
            performance_data[model_name]['recall'],
            performance_data[model_name]['f1_score'],
            performance_data[model_name]['roc_auc'] or 0,
            normalized_lb
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model_name.upper(),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart",
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Add explanation for leaderboard score
    st.info("ℹ️ **Leaderboard Score**: This is the custom German Credit competition metric that considers loan amounts and real-world costs. Lower scores are better, so it's inverted in the radar chart for visualization.")
    
    # Best model highlighting based on leaderboard score (lower is better)
    best_model_name = min(selected_models, key=lambda x: performance_data[x]['leaderboard_score'])
    best_score = performance_data[best_model_name]['leaderboard_score']
    st.success(f"🥇 **Best Performing Model (Leaderboard)**: {best_model_name.upper()} with Leaderboard Score: {best_score:.4f} (lower is better)")


def show_detailed_metrics(model_data, selected_models):
    """Show detailed evaluation metrics"""
    st.markdown("### 📊 Detailed Performance Metrics")
    
    # Calculate detailed metrics for each model
    detailed_metrics = {}
    
    with st.spinner("Calculating detailed metrics..."):
        for model_name in selected_models:
            model = model_data['models'][model_name]
            metrics = calculate_comprehensive_metrics(model, model_data['X'], model_data['y'])
            detailed_metrics[model_name] = metrics
    
    # Create detailed comparison table
    st.markdown("#### 📋 Credit Risk Specific Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Risk Detection Metrics**")
        risk_metrics = []
        for model_name in selected_models:
            metrics = detailed_metrics[model_name]
            risk_metrics.append({
                'Model': model_name.upper(),
                'Sensitivity (Recall)': f"{metrics['sensitivity']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'PPV (Precision)': f"{metrics['ppv']:.4f}",
                'NPV': f"{metrics['npv']:.4f}"
            })
        
        risk_df = pd.DataFrame(risk_metrics)
        st.dataframe(risk_df, width='stretch')
    
    with col2:
        st.markdown("**📊 Overall Performance**")
        overall_metrics = []
        for model_name in selected_models:
            metrics = detailed_metrics[model_name]
            overall_metrics.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Balanced Accuracy': f"{metrics['balanced_accuracy']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
            })
        
        overall_df = pd.DataFrame(overall_metrics)
        st.dataframe(overall_df, width='stretch')
    
    # Model comparison charts
    st.markdown("#### 📈 Metric Comparison Charts")
    
    # Accuracy comparison
    acc_data = {
        'Model': [name.upper() for name in selected_models],
        'Accuracy': [detailed_metrics[name]['accuracy'] for name in selected_models],
        'Balanced Accuracy': [detailed_metrics[name]['balanced_accuracy'] for name in selected_models],
        'Cost-Weighted Accuracy': [detailed_metrics[name]['cost_weighted_accuracy'] for name in selected_models]
    }
    
    acc_df = pd.DataFrame(acc_data)
    
    fig = px.bar(
        acc_df, 
        x='Model', 
        y=['Accuracy', 'Balanced Accuracy', 'Cost-Weighted Accuracy'],
        title="Accuracy Metrics Comparison",
        barmode='group'
    )
    st.plotly_chart(fig, width='stretch')


def show_roc_analysis(model_data, selected_models):
    """Show ROC curve analysis"""
    st.markdown("### 📈 ROC Curve Analysis")
    
    try:
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_curve, auc
        
        fig = go.Figure()
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        for model_name in selected_models:
            if model_name == 'ensemble':
                model = model_data['ensemble']
                display_name = 'Ensemble'
            else:
                model = model_data['models'][model_name]
                display_name = model_name.upper()
            
            # Get prediction probabilities
            y_proba = cross_val_predict(
                model, model_data['X'], model_data['y'], 
                cv=5, method='predict_proba'
            )[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(model_data['y'], y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Add to plot
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{display_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.info("💡 **ROC Curve Interpretation**: The closer the curve is to the top-left corner, the better the model performance. AUC values closer to 1.0 indicate better classification ability.")
        
    except Exception as e:
        st.error(f"Error generating ROC curves: {str(e)}")


def show_confusion_matrix_analysis(model_data, selected_models):
    """Show confusion matrix analysis"""
    st.markdown("### 🧮 Confusion Matrix Analysis")
    
    try:
        from sklearn.model_selection import cross_val_predict
        
        cols = st.columns(min(len(selected_models), 3))
        
        for i, model_name in enumerate(selected_models):
            with cols[i % 3]:
                if model_name == 'ensemble':
                    model = model_data['ensemble']
                    display_name = 'Ensemble'
                else:
                    model = model_data['models'][model_name]
                    display_name = model_name.upper()
                
                # Get cross-validated predictions
                y_pred = cross_val_predict(model, model_data['X'], model_data['y'], cv=5)
                
                # Calculate confusion matrix
                cm = confusion_matrix(model_data['y'], y_pred)
                
                # Create heatmap
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f'{display_name} Confusion Matrix',
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Bad Credit', 'Good Credit'],
                    y=['Bad Credit', 'Good Credit'],
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Calculate metrics from confusion matrix
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                st.markdown(f"""
                **{display_name} Metrics:**
                - Accuracy: {accuracy:.3f}
                - Precision: {precision:.3f}
                - Recall: {recall:.3f}
                - Specificity: {specificity:.3f}
                """)
        
    except Exception as e:
        st.error(f"Error generating confusion matrices: {str(e)}")


def show_classification_report_analysis(model_data, selected_models):
    """Show detailed classification reports"""
    st.markdown("### 📋 Classification Reports")
    
    try:
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import classification_report
        
        for model_name in selected_models:
            if model_name == 'ensemble':
                model = model_data['ensemble']
                display_name = 'Ensemble'
            else:
                model = model_data['models'][model_name]
                display_name = model_name.upper()
            
            st.markdown(f"#### 📊 {display_name} Classification Report")
            
            # Get cross-validated predictions
            y_pred = cross_val_predict(model, model_data['X'], model_data['y'], cv=5)
            
            # Generate classification report
            report = classification_report(
                model_data['y'], 
                y_pred, 
                target_names=['Bad Credit', 'Good Credit'],
                output_dict=True
            )
            
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            
            # Style and display
            st.dataframe(
                report_df.style.format({
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'f1-score': '{:.3f}',
                    'support': '{:.0f}'
                }),
                width='stretch'
            )
            
            st.markdown("---")
    
    except Exception as e:
        st.error(f"Error generating classification reports: {str(e)}")


def get_model_type(model_name):
    """Get descriptive model type"""
    model_types = {
        'svc': 'Support Vector Classifier',
        'xgb': 'XGBoost Classifier',
        'rf': 'Random Forest',
        'histgbc': 'Histogram Gradient Boosting'
    }
    return model_types.get(model_name, 'Unknown')