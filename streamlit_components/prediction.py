"""
Prediction Components for Streamlit App
=======================================

This module contains a single prediction function
for the German Credit Risk Analysis Streamlit application.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')


def predict_test_data_and_save(model_name='xgb'):
    """
    Predict on data/german_credit_test.csv and save results to 
    data/german_credit_test_submission_{model_name}.csv
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use ('xgb' or 'svc')
    
    Returns:
    --------
    str : Path to the saved submission file
    """
    try:
        from src.utils.data_loader import load_german_credit_data, prepare_features_and_target
        from src.utils.data_preprocessing import apply_feature_transformations
        from src.model.best_models import create_best_models
        
        # Load and prepare data
        trainset, testset = load_german_credit_data()
        trainset_transformed, testset_transformed = apply_feature_transformations(trainset, testset)
        X, y, test_X = prepare_features_and_target(trainset_transformed, testset_transformed)
        
        # Create and train the specified model
        models = create_best_models(X)
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
        
        model = models[model_name]
        model.fit(X, y)
        
        # Make predictions on test data
        predictions = model.predict(test_X)
        
        # Convert predictions to Risk/No Risk format
        # Assuming 1 = No Risk, 0 = Risk based on typical binary classification
        risk_predictions = ['No Risk' if pred == 1 else 'Risk' for pred in predictions]
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'Id': range(len(predictions)),
            'Risk': risk_predictions
        })
        
        # Save to file with model name
        output_file = f'data/german_credit_test_submission_{model_name}.csv'
        submission_df.to_csv(output_file, index=False)
        
        print(f"Predictions saved to {output_file}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Risk predictions: {sum(1 for p in risk_predictions if p == 'Risk')}")
        print(f"No Risk predictions: {sum(1 for p in risk_predictions if p == 'No Risk')}")
        
        return output_file
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None


# For backward compatibility with the streamlit app
def show_prediction_page():
    """Simplified prediction page that just runs the prediction function"""
    import streamlit as st
    
    st.markdown('<h2 class="section-header">🎯 Generate Test Predictions</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 💡 Generate Kaggle Submission File
    This will predict on `data/german_credit_test.csv` and save results to a submission file.
    """)
    
    # Model selection
    model_name = st.selectbox(
        "Choose model for predictions:",
        ['xgb', 'svc'],
        format_func=lambda x: 'XGBoost' if x == 'xgb' else 'Support Vector Classifier'
    )
    
    # Prediction button
    if st.button("🔮 Generate Test Predictions", type="primary"):
        with st.spinner("Making predictions on test data..."):
            output_file = predict_test_data_and_save(model_name)
            
            if output_file:
                st.success(f"✅ Predictions saved to `{output_file}`")
                
                # Show download button
                try:
                    with open(output_file, 'r') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Submission File",
                        data=csv_data,
                        file_name=f"german_credit_test_submission_{model_name}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error reading file for download: {str(e)}")
            else:
                st.error("❌ Failed to generate predictions")