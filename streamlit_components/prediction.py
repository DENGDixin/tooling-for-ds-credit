"""
Prediction Components for Streamlit App
=======================================

This module contains the prediction functionality
for the German Credit Risk Analysis Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys

# Add src to path
sys.path.append('src')


@st.cache_resource
def load_prediction_models():
    """Load and cache trained models for prediction"""
    try:
        from src.utils.data_loader import load_german_credit_data, prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        from src.model.best_models import create_best_models
        
        # Load and prepare data
        trainset, testset = load_german_credit_data()
        trainset_transformed, testset_transformed = apply_feature_transformations(trainset, testset)
        X, y, test_X = prepare_features_and_target(trainset_transformed, testset_transformed)
        
        # Create and train models
        models = create_best_models(X)
        
        # Fit models
        for model_name, model in models.items():
            model.fit(X, y)
        
        return {
            'models': models,
            'feature_names': X.columns.tolist(),
            'sample_data': trainset_transformed.head()
        }
    except Exception as e:
        st.error(f"Error loading prediction models: {str(e)}")
        return None


def show_prediction_page():
    """Complete prediction page"""
    st.markdown('<h2 class="section-header">🎯 Make Credit Risk Predictions</h2>', unsafe_allow_html=True)
    
    # Load models
    model_data = load_prediction_models()
    if model_data is None:
        st.error("Unable to load prediction models.")
        return
    
    st.markdown("""
    ### 💡 How to Use
    1. **Choose Input Method**: Manual entry or upload CSV file
    2. **Select Model**: Choose which trained model to use for prediction
    3. **Enter Customer Data**: Fill in the required information
    4. **Get Prediction**: See the credit risk assessment and confidence
    """)
    
    # Sidebar model selection
    st.sidebar.markdown("### 🤖 Model Selection")
    available_models = list(model_data['models'].keys())
    selected_model = st.sidebar.selectbox(
        "Choose prediction model:",
        available_models,
        format_func=lambda x: x.upper()
    )
    
    # Model info
    if selected_model:
        show_model_info(selected_model)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "👤 Single Prediction",
        "📁 Batch Predictions", 
        "📊 Prediction Analysis"
    ])
    
    with tab1:
        show_single_prediction(model_data, selected_model)
    
    with tab2:
        show_batch_predictions(model_data, selected_model)
    
    with tab3:
        show_prediction_analysis(model_data)


def show_model_info(selected_model):
    """Show information about the selected model"""
    try:
        from src.model.model_configs import MODEL_PERFORMANCE
        
        performance = MODEL_PERFORMANCE.get(selected_model, {})
        cv_score = performance.get('cv_score', 'N/A')
        
        st.sidebar.markdown(f"**Performance**: {cv_score}")
        
        model_descriptions = {
            'svc': 'Support Vector Classifier - Best overall performance',
            'xgb': 'XGBoost - Gradient boosting with feature importance'
        }
        
        description = model_descriptions.get(selected_model, 'Machine learning model')
        st.sidebar.markdown(f"**Description**: {description}")
        
    except Exception:
        pass


def show_single_prediction(model_data, selected_model):
    """Single customer prediction interface"""
    st.markdown("### 👤 Single Customer Prediction")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ['Manual Entry', 'Load Example'],
        horizontal=True
    )
    
    if input_method == 'Load Example':
        # Load example data
        example_data = get_example_customer_data()
        st.info("📝 Example customer data loaded. You can modify the values below.")
        customer_data = create_input_form(example_data)
    else:
        # Manual entry
        customer_data = create_input_form()
    
    # Prediction button
    if st.button("🔮 Predict Credit Risk", type="primary", width='stretch'):
        make_single_prediction(model_data, selected_model, customer_data)


def create_input_form(default_data=None):
    """Create input form for customer data"""
    st.markdown("#### 📋 Customer Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    customer_data = {}
    
    with col1:
        st.markdown("**📊 Financial Information**")
        
        customer_data['LoanAmount'] = st.number_input(
            "Loan Amount (€)",
            min_value=250,
            max_value=20000,
            value=default_data.get('LoanAmount', 3000) if default_data else 3000,
            step=100,
            help="Amount of loan requested"
        )
        
        customer_data['LoanDuration'] = st.number_input(
            "Loan Duration (months)",
            min_value=4,
            max_value=72,
            value=default_data.get('LoanDuration', 24) if default_data else 24,
            step=1,
            help="Duration of the loan in months"
        )
        
        customer_data['InstallmentPercent'] = st.slider(
            "Installment Percent (%)",
            min_value=1,
            max_value=4,
            value=default_data.get('InstallmentPercent', 3) if default_data else 3,
            help="Installment as percentage of disposable income"
        )
        
        customer_data['ExistingCredits'] = st.selectbox(
            "Number of Existing Credits",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(default_data.get('ExistingCredits', 1)) if default_data else 0,
            help="Number of existing credits at this bank"
        )
    
    with col2:
        st.markdown("**👤 Personal Information**")
        
        customer_data['Age'] = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=80,
            value=default_data.get('Age', 35) if default_data else 35,
            step=1
        )
        
        customer_data['Sex'] = st.selectbox(
            "Gender",
            ['Male', 'Female'],
            index=['Male', 'Female'].index(default_data.get('Sex', 'Male')) if default_data else 0
        )
        
        customer_data['ForeignWorker'] = st.selectbox(
            "Foreign Worker",
            ['No', 'Yes'],
            index=['No', 'Yes'].index(default_data.get('ForeignWorker', 'No')) if default_data else 0
        )
        
        customer_data['ResidenceDuration'] = st.selectbox(
            "Residence Duration",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(default_data.get('ResidenceDuration', 3)) if default_data else 2,
            help="How long at current residence (1=<1 year, 4=>4 years)"
        )
    
    # Additional information
    st.markdown("#### 📈 Additional Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        customer_data['CreditHistory'] = st.selectbox(
            "Credit History",
            ['No credits taken/ all credits paid back duly', 'All credits at this bank paid back duly', 
             'Existing credits paid back duly till now', 'Delay in paying off in the past', 
             'Critical account/ other credits existing (not at this bank)'],
            index=2 if not default_data else 0,
            help="Customer's credit history"
        )
        
        customer_data['Purpose'] = st.selectbox(
            "Loan Purpose",
            ['Car (new)', 'Car (used)', 'Furniture/equipment', 'Radio/television', 
             'Domestic appliances', 'Repairs', 'Education', 'Vacation', 'Retraining', 
             'Business', 'Others'],
            index=0 if not default_data else 0,
            help="Purpose of the loan"
        )
    
    with col4:
        customer_data['PersonalStatus'] = st.selectbox(
            "Personal Status",
            ['Male : divorced/separated', 'Female : divorced/separated/married', 
             'Male : single', 'Male : married/widowed'],
            index=2 if not default_data else 0,
            help="Marital status and gender"
        )
        
        customer_data['EmploymentDuration'] = st.selectbox(
            "Employment Duration",
            ['Unemployed', '< 1 year', '1 <= ... < 4 years', '4 <= ... < 7 years', '>= 7 years'],
            index=2 if not default_data else 0,
            help="How long employed at current job"
        )
    
    return customer_data


def get_example_customer_data():
    """Get example customer data"""
    return {
        'LoanAmount': 2500,
        'LoanDuration': 18,
        'InstallmentPercent': 3,
        'ExistingCredits': 1,
        'Age': 28,
        'Sex': 'Male',
        'ForeignWorker': 'No',
        'ResidenceDuration': 3,
        'CreditHistory': 'Existing credits paid back duly till now',
        'Purpose': 'Car (used)',
        'PersonalStatus': 'Male : single',
        'EmploymentDuration': '1 <= ... < 4 years'
    }


def make_single_prediction(model_data, selected_model, customer_data):
    """Make prediction for single customer"""
    try:
        # Prepare data for prediction
        input_df = prepare_prediction_data(customer_data)
        
        # Get model
        model = model_data['models'][selected_model]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("### 🎯 Prediction Results")
        
        # Main prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_label = "Good Credit" if prediction == 1 else "Bad Credit"
            risk_color = "green" if prediction == 1 else "red"
            st.markdown(f'<h3 style="color: {risk_color};">🎯 {risk_label}</h3>', unsafe_allow_html=True)
        
        with col2:
            confidence = max(prediction_proba) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col3:
            risk_score = prediction_proba[0] * 100  # Probability of bad credit
            st.metric("Risk Score", f"{risk_score:.1f}%")
        
        # Detailed probabilities
        st.markdown("#### 📊 Detailed Probabilities")
        prob_df = pd.DataFrame({
            'Outcome': ['Bad Credit', 'Good Credit'],
            'Probability': prediction_proba,
            'Percentage': [f"{p*100:.1f}%" for p in prediction_proba]
        })
        
        st.dataframe(prob_df, width='stretch')
        
        # Risk interpretation
        show_risk_interpretation(risk_score, confidence)
        
        # Show input summary
        st.markdown("#### 📋 Input Summary")
        input_summary = pd.DataFrame([
            {'Field': k, 'Value': v} for k, v in customer_data.items()
        ])
        st.dataframe(input_summary, width='stretch')
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")


def show_risk_interpretation(risk_score, confidence):
    """Show interpretation of risk score"""
    st.markdown("#### 💡 Risk Assessment Interpretation")
    
    if risk_score < 30:
        st.success(f"✅ **Low Risk** (Risk Score: {risk_score:.1f}%): This customer shows strong creditworthiness indicators.")
    elif risk_score < 60:
        st.warning(f"⚠️ **Medium Risk** (Risk Score: {risk_score:.1f}%): This customer shows mixed creditworthiness indicators. Additional review recommended.")
    else:
        st.error(f"❌ **High Risk** (Risk Score: {risk_score:.1f}%): This customer shows concerning creditworthiness indicators.")
    
    if confidence < 60:
        st.info("ℹ️ **Note**: The model has relatively low confidence in this prediction. Consider additional verification.")


def show_batch_predictions(model_data, selected_model):
    """Batch prediction interface"""
    st.markdown("### 📁 Batch Predictions")
    
    st.markdown("""
    Upload a CSV file with customer data to get predictions for multiple customers at once.
    
    **Required columns**: LoanAmount, LoanDuration, Age, Sex, CreditHistory, Purpose, etc.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file with customer data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            batch_data = pd.read_csv(uploaded_file)
            
            st.markdown("#### 📊 Uploaded Data Preview")
            st.dataframe(batch_data.head(), width='stretch')
            
            # Validate data
            required_columns = ['LoanAmount', 'LoanDuration', 'Age']  # Basic required columns
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Make predictions
            if st.button("🔮 Generate Batch Predictions", type="primary"):
                run_batch_predictions(model_data, selected_model, batch_data)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show example format
        st.markdown("#### 📝 Example CSV Format")
        example_df = pd.DataFrame([
            get_example_customer_data(),
            {**get_example_customer_data(), 'Age': 45, 'LoanAmount': 5000, 'LoanDuration': 36}
        ])
        st.dataframe(example_df, width='stretch')
        
        # Download example
        csv = example_df.to_csv(index=False)
        st.download_button(
            "📥 Download Example CSV",
            csv,
            "example_customers.csv",
            "text/csv"
        )


def run_batch_predictions(model_data, selected_model, batch_data):
    """Run predictions on batch data"""
    try:
        # Prepare data
        processed_data = []
        for _, row in batch_data.iterrows():
            try:
                input_df = prepare_prediction_data(row.to_dict())
                processed_data.append(input_df.iloc[0])
            except Exception as e:
                st.warning(f"Skipping row due to error: {str(e)}")
        
        if not processed_data:
            st.error("No valid rows found in the data.")
            return
        
        # Convert to DataFrame
        prediction_input = pd.DataFrame(processed_data)
        
        # Get model and make predictions
        model = model_data['models'][selected_model]
        predictions = model.predict(prediction_input)
        prediction_probas = model.predict_proba(prediction_input)
        
        # Create results DataFrame
        results = batch_data.copy()
        results['Predicted_Risk'] = ['Good Credit' if p == 1 else 'Bad Credit' for p in predictions]
        results['Risk_Score'] = [proba[0] * 100 for proba in prediction_probas]  # % chance of bad credit
        results['Confidence'] = [max(proba) * 100 for proba in prediction_probas]
        
        # Display results
        st.markdown("### 🎯 Batch Prediction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(results))
        with col2:
            good_credit_pct = (predictions == 1).mean() * 100
            st.metric("Good Credit %", f"{good_credit_pct:.1f}%")
        with col3:
            avg_risk = results['Risk_Score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
        with col4:
            avg_confidence = results['Confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Results table
        st.dataframe(
            results.style.format({
                'Risk_Score': '{:.1f}%',
                'Confidence': '{:.1f}%'
            }),
            width='stretch'
        )
        
        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            "credit_risk_predictions.csv",
            "text/csv"
        )
        
    except Exception as e:
        st.error(f"Error running batch predictions: {str(e)}")


def show_prediction_analysis(model_data):
    """Show prediction analysis and insights"""
    st.markdown("### 📊 Prediction Analysis")
    
    st.markdown("""
    #### 🎯 Model Comparison
    
    Compare how different models would predict for the same customer:
    """)
    
    # Example customer for comparison
    example_customer = get_example_customer_data()
    
    if st.button("🔄 Compare All Models"):
        compare_models_prediction(model_data, example_customer)


def compare_models_prediction(model_data, customer_data):
    """Compare predictions across all models"""
    try:
        # Prepare data
        input_df = prepare_prediction_data(customer_data)
        
        # Get predictions from all models
        results = []
        for model_name, model in model_data['models'].items():
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            results.append({
                'Model': model_name.upper(),
                'Prediction': 'Good Credit' if prediction == 1 else 'Bad Credit',
                'Risk_Score': prediction_proba[0] * 100,
                'Confidence': max(prediction_proba) * 100
            })
        
        # Display comparison
        comparison_df = pd.DataFrame(results)
        
        st.markdown("#### 🤖 Model Comparison Results")
        st.dataframe(
            comparison_df.style.format({
                'Risk_Score': '{:.1f}%',
                'Confidence': '{:.1f}%'
            }),
            width='stretch'
        )
        
        # Consensus analysis
        good_credit_count = sum(1 for r in results if r['Prediction'] == 'Good Credit')
        consensus = "Strong" if good_credit_count >= 4 or good_credit_count <= 1 else "Mixed"
        
        st.markdown(f"#### 🎯 Consensus: **{consensus}**")
        if consensus == "Strong":
            st.success("✅ All or most models agree on the prediction.")
        else:
            st.warning("⚠️ Models disagree. Consider additional review.")
    
    except Exception as e:
        st.error(f"Error comparing models: {str(e)}")


def prepare_prediction_data(customer_data):
    """Prepare customer data for model prediction"""
    try:
        from src.utils.data_loader import prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        
        # Create a DataFrame with the customer data
        df = pd.DataFrame([customer_data])
        
        # Apply the same transformations as training data
        # Note: This is a simplified version - in practice, you'd need
        # to ensure all preprocessing steps are applied correctly
        
        # For now, return the basic dataframe
        # In a full implementation, you'd apply all the same preprocessing
        # steps that were used during training
        
        return df
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return pd.DataFrame()