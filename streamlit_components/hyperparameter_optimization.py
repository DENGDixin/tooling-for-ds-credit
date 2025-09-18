"""
Hyperparameter Optimization Components for Streamlit App
=======================================================

This module contains the hyperparameter optimization functionality
for the German Credit Risk Analysis Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Add src to path
sys.path.append('src')


def show_hyperparameter_optimization_page():
    """Complete hyperparameter optimization page"""
    st.markdown('<h2 class="section-header">⚙️ Hyperparameter Optimization</h2>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ### 🎯 Optimize Model Performance
    
    Use this page to fine-tune hyperparameters for different machine learning models. 
    The optimization uses **Optuna** for efficient hyperparameter search.
    """)
    
    # Load data first
    data = load_optimization_data()
    if data is None:
        st.error("Unable to load data for optimization.")
        return
    
    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Optimization Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model to Optimize:",
        ['SVC', 'XGBoost'],
        help="Choose which model to optimize"
    )
    
    # Optimization settings
    n_trials = st.sidebar.slider(
        "Number of Trials:",
        min_value=5,
        max_value=200,
        value=20,
        help="More trials = better optimization but takes longer"
    )
    
    optimization_mode = st.sidebar.radio(
        "Optimization Mode:",
        ['Quick Demo', 'Standard', 'Thorough'],
        help="Demo: 5-10 trials, Standard: 20-50 trials, Thorough: 100+ trials"
    )
    
    # Adjust trials based on mode
    if optimization_mode == 'Quick Demo':
        n_trials = min(n_trials, 10)
    elif optimization_mode == 'Thorough':
        n_trials = max(n_trials, 50)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "🚀 Run Optimization",
        "📊 Current Best Parameters", 
        "📈 Optimization History"
    ])
    
    with tab1:
        show_run_optimization(data, model_type, n_trials, optimization_mode)
    
    with tab2:
        show_current_best_parameters()
    
    with tab3:
        show_optimization_history()


@st.cache_data
def load_optimization_data():
    """Load data for optimization"""
    try:
        from src.utils.data_loader import load_german_credit_data, prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        
        # Load and prepare data
        trainset, testset = load_german_credit_data()
        trainset_transformed, testset_transformed = apply_feature_transformations(trainset, testset)
        X, y, test_X = prepare_features_and_target(trainset_transformed, testset_transformed)
        
        return {'X': X, 'y': y}
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def show_run_optimization(data, model_type, n_trials, optimization_mode):
    """Run hyperparameter optimization"""
    st.markdown("### 🚀 Run Hyperparameter Optimization")
    
    # Show current settings
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", model_type)
    with col2:
        st.metric("Trials", n_trials)
    with col3:
        st.metric("Mode", optimization_mode)
    
    # Time estimation
    estimated_time = estimate_optimization_time(model_type, n_trials)
    st.info(f"⏱️ Estimated time: {estimated_time}")
    
    # Warning for long optimizations
    if n_trials > 50:
        st.warning("⚠️ This optimization may take a while. Consider starting with fewer trials.")
    
    # Run optimization button
    if st.button(f"🚀 Start {model_type} Optimization", type="primary", width='stretch'):
        run_model_optimization(data, model_type, n_trials)


def run_model_optimization(data, model_type, n_trials):
    """Execute the optimization process"""
    try:
        # Import optimization functions
        from src.model.hyperparameter_optimization import (
            create_objective_svc,
            create_objective_xgb
        )
        import optuna
        
        # Suppress optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Select objective function
        if model_type == 'SVC':
            objective = create_objective_svc(data['X'], data['y'])
        elif model_type == 'XGBoost':
            objective = create_objective_xgb(data['X'], data['y'])
        
        # Custom callback for progress tracking
        def progress_callback(study, trial):
            progress = trial.number / n_trials
            progress_bar.progress(progress)
            status_text.text(f"Trial {trial.number + 1}/{n_trials} - Current best: {study.best_value:.4f}")
        
        # Run optimization
        status_text.text(f"Starting {model_type} optimization...")
        start_time = time.time()
        
        study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Show results
        progress_bar.progress(1.0)
        status_text.text("✅ Optimization completed!")
        
        show_optimization_results(study, model_type, optimization_time, n_trials)
        
        # Store results in session state for history
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []
        
        st.session_state.optimization_history.append({
            'model': model_type,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials,
            'time': optimization_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        st.error(f"Error during optimization: {str(e)}")


def show_optimization_results(study, model_type, optimization_time, n_trials):
    """Display optimization results"""
    st.markdown("### 🎉 Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Score", f"{study.best_value:.4f}")
    with col2:
        st.metric("Trials Completed", n_trials)
    with col3:
        st.metric("Time Taken", f"{optimization_time:.1f}s")
    with col4:
        improvement = ((study.best_value - study.trials[0].value) / study.trials[0].value) * 100
        st.metric("Improvement", f"{improvement:.1f}%")
    
    # Best parameters
    st.markdown("#### 🏆 Best Parameters Found")
    params_df = pd.DataFrame([
        {'Parameter': k, 'Value': v} 
        for k, v in study.best_params.items()
    ])
    st.dataframe(params_df, width='stretch')
    
    # Optimization history chart
    st.markdown("#### 📈 Optimization Progress")
    
    # Extract trial data
    trial_numbers = [trial.number for trial in study.trials]
    trial_values = [trial.value for trial in study.trials]
    best_values = []
    current_best = float('-inf')
    
    for value in trial_values:
        if value > current_best:
            current_best = value
        best_values.append(current_best)
    
    # Create progress chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trial_numbers, trial_values, 'o-', alpha=0.7, label='Trial Score')
    ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Score So Far')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_type} Optimization Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Parameter importance (if enough trials)
    if len(study.trials) >= 10:
        show_parameter_importance(study)


def show_parameter_importance(study):
    """Show parameter importance analysis"""
    try:
        import optuna
        
        st.markdown("#### 🎯 Parameter Importance")
        
        # Calculate parameter importance
        importance = optuna.importance.get_param_importances(study)
        
        if importance:
            # Create importance dataframe
            importance_df = pd.DataFrame([
                {'Parameter': k, 'Importance': v} 
                for k, v in importance.items()
            ]).sort_values('Importance', ascending=False)
            
            # Display chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(importance_df)), importance_df['Importance'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['Parameter'])
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
            ax.invert_yaxis()
            
            # Color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show table
            st.dataframe(
                importance_df.style.format({'Importance': '{:.4f}'}),
                width='stretch'
            )
    
    except Exception as e:
        st.warning(f"Could not calculate parameter importance: {str(e)}")


def show_current_best_parameters():
    """Show current best parameters from model configs"""
    st.markdown("### 📊 Current Best Parameters")
    
    try:
        from src.model.model_configs import BEST_MODEL_PARAMS, MODEL_PERFORMANCE
        
        # Model selection
        model_names = list(BEST_MODEL_PARAMS.keys())
        selected_model = st.selectbox("Select model to view:", model_names)
        
        if selected_model:
            # Show performance
            performance = MODEL_PERFORMANCE.get(selected_model, {})
            cv_score = performance.get('cv_score', 0)
            
            st.metric(f"{selected_model.upper()} Performance", f"{cv_score:.4f}")
            
            # Show parameters
            params = BEST_MODEL_PARAMS[selected_model]
            
            st.markdown(f"#### 🎛️ {selected_model.upper()} Parameters")
            
            params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v, 'Type': type(v).__name__} 
                for k, v in params.items()
            ])
            
            st.dataframe(params_df, width='stretch')
            
            # Copy parameters button
            if st.button(f"📋 Copy {selected_model.upper()} Parameters"):
                params_str = str(params)
                st.code(params_str, language='python')
                st.success("Parameters copied! You can use these in your code.")
    
    except Exception as e:
        st.error(f"Error loading current parameters: {str(e)}")


def show_optimization_history():
    """Show optimization history from session"""
    st.markdown("### 📈 Optimization History")
    
    if 'optimization_history' in st.session_state and st.session_state.optimization_history:
        history = st.session_state.optimization_history
        
        # Create history dataframe
        history_df = pd.DataFrame(history)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Optimizations", len(history))
        with col2:
            best_overall = max(history, key=lambda x: x['best_score'])
            st.metric("Best Overall Score", f"{best_overall['best_score']:.4f}")
        with col3:
            st.metric("Best Model", best_overall['model'])
        
        # History table
        st.markdown("#### 📋 Optimization Runs")
        display_df = history_df[['timestamp', 'model', 'best_score', 'n_trials', 'time']].copy()
        display_df['time'] = display_df['time'].round(1)
        
        st.dataframe(
            display_df.style.format({
                'best_score': '{:.4f}',
                'time': '{:.1f}s'
            }),
            width='stretch'
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.optimization_history = []
            st.experimental_rerun()
    
    else:
        st.info("No optimization history yet. Run some optimizations to see results here!")


def estimate_optimization_time(model_type, n_trials):
    """Estimate optimization time"""
    # Time estimates per trial (in seconds)
    time_per_trial = {
        'SVC': 3,
        'XGBoost': 2
    }
    
    base_time = time_per_trial.get(model_type, 3)
    total_seconds = base_time * n_trials
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        return f"{total_seconds/3600:.1f} hours"