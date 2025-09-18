"""
Unit tests for data_loader.py module using pytest
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import load_german_credit_data, prepare_features_and_target


class TestLoadGermanCreditData:
    """Test class for load_german_credit_data function"""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create temporary data directory with sample CSV files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample training data
            train_data = pd.DataFrame({
                'Risk': ['Risk', 'No Risk', 'Risk', 'No Risk'],
                'LoanAmount': [1000, 2000, 1500, 3000],
                'LoanDuration': [12, 24, 18, 36],
                'Age': [25, 35, 45, 55],
                'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent'],
                'Purpose': ['Car', 'House', 'Education', 'Business']
            })
            
            # Create sample test data
            test_data = pd.DataFrame({
                'Id': [1, 2, 3, 4],
                'LoanAmount': [1200, 1800, 2500, 3500],
                'LoanDuration': [15, 20, 30, 40],
                'Age': [28, 38, 48, 58],
                'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent'],
                'Purpose': ['Car', 'House', 'Education', 'Business']
            })
            
            # Save to temporary directory
            train_path = Path(temp_dir) / 'german_credit_train.csv'
            test_path = Path(temp_dir) / 'german_credit_test.csv'
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            yield temp_dir
    
    def test_load_german_credit_data_success(self, sample_data_dir):
        """Test successful loading of German credit data"""
        trainset, testset = load_german_credit_data(sample_data_dir)
        
        # Check that both datasets are DataFrames
        assert isinstance(trainset, pd.DataFrame)
        assert isinstance(testset, pd.DataFrame)
        
        # Check expected columns
        expected_train_cols = ['Risk', 'LoanAmount', 'LoanDuration', 'Age', 'CreditHistory', 'Purpose']
        expected_test_cols = ['Id', 'LoanAmount', 'LoanDuration', 'Age', 'CreditHistory', 'Purpose']
        
        assert list(trainset.columns) == expected_train_cols
        assert list(testset.columns) == expected_test_cols
        
        # Check data shapes
        assert trainset.shape == (4, 6)
        assert testset.shape == (4, 6)
    
    def test_load_german_credit_data_file_not_found(self):
        """Test behavior when data files don't exist"""
        with pytest.raises(FileNotFoundError):
            load_german_credit_data("nonexistent_directory")
    
    def test_load_german_credit_data_custom_directory(self, sample_data_dir):
        """Test loading with custom data directory"""
        trainset, testset = load_german_credit_data(sample_data_dir)
        
        assert not trainset.empty
        assert not testset.empty
        assert 'Risk' in trainset.columns
        assert 'Id' in testset.columns


class TestPrepareFeaturesAndTarget:
    """Test class for prepare_features_and_target function"""
    
    @pytest.fixture
    def sample_datasets(self):
        """Create sample training and test datasets"""
        trainset = pd.DataFrame({
            'Risk': ['Risk', 'No Risk', 'Risk', 'No Risk', 'Risk'],
            'LoanAmount': [1000, 2000, 1500, 3000, 2500],
            'LoanDuration': [12, 24, 18, 36, 30],
            'Age': [25, 35, 45, 55, 40],
            'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent', 'Good'],
            'Purpose': ['Car', 'House', 'Education', 'Business', 'Car'],
            'Employment': ['Skilled', 'Unskilled', 'Skilled', 'Management', 'Skilled']
        })
        
        testset = pd.DataFrame({
            'Id': [1, 2, 3],
            'LoanAmount': [1200, 1800, 2500],
            'LoanDuration': [15, 20, 30],
            'Age': [28, 38, 48],
            'CreditHistory': ['Good', 'Bad', 'Good'],
            'Purpose': ['Car', 'House', 'Education'],
            'Employment': ['Skilled', 'Unskilled', 'Management']
        })
        
        return trainset, testset
    
    def test_prepare_features_and_target_basic(self, sample_datasets):
        """Test basic functionality of prepare_features_and_target"""
        trainset, testset = sample_datasets
        X, y, test_X = prepare_features_and_target(trainset, testset)
        
        # Check return types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(test_X, pd.DataFrame)
        
        # Check shapes
        assert X.shape == (5, 6)  # 5 rows, 6 features (excluding Risk)
        assert y.shape == (5,)    # 5 target values
        assert test_X.shape == (3, 6)  # 3 rows, 6 features (excluding Id)
        
        # Check that Risk column is removed from X
        assert 'Risk' not in X.columns
        
        # Check that Id column is removed from test_X
        assert 'Id' not in test_X.columns
    
    def test_risk_column_mapping(self, sample_datasets):
        """Test that Risk column is correctly mapped to binary values"""
        trainset, testset = sample_datasets
        X, y, test_X = prepare_features_and_target(trainset, testset)
        
        # Check binary mapping
        assert set(y.unique()) == {0, 1}
        
        # Check specific mappings
        original_risk = trainset['Risk'].tolist()
        expected_y = [1 if risk == 'Risk' else 0 for risk in original_risk]
        assert y.tolist() == expected_y
    
    def test_categorical_conversion(self, sample_datasets):
        """Test that categorical columns are properly converted"""
        trainset, testset = sample_datasets
        X, y, test_X = prepare_features_and_target(trainset, testset)
        
        # Check that object columns are converted to category
        object_columns = ['CreditHistory', 'Purpose', 'Employment']
        
        for col in object_columns:
            assert X[col].dtype.name == 'category'
            assert test_X[col].dtype.name == 'category'
    
    def test_feature_consistency(self, sample_datasets):
        """Test that X and test_X have same columns (except target)"""
        trainset, testset = sample_datasets
        X, y, test_X = prepare_features_and_target(trainset, testset)
        
        # Check that feature columns are the same
        assert list(X.columns) == list(test_X.columns)
        
        # Check that all expected features are present
        expected_features = ['LoanAmount', 'LoanDuration', 'Age', 'CreditHistory', 'Purpose', 'Employment']
        assert list(X.columns) == expected_features
    
    def test_original_data_unchanged(self, sample_datasets):
        """Test that original datasets are not modified"""
        trainset, testset = sample_datasets
        original_trainset = trainset.copy()
        original_testset = testset.copy()
        
        X, y, test_X = prepare_features_and_target(trainset, testset)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(trainset, original_trainset)
        pd.testing.assert_frame_equal(testset, original_testset)
    
    def test_empty_dataframe_handling(self):
        """Test behavior with empty DataFrames"""
        empty_train = pd.DataFrame({'Risk': []})
        empty_test = pd.DataFrame({'Id': []})
        
        X, y, test_X = prepare_features_and_target(empty_train, empty_test)
        
        assert X.empty
        assert y.empty
        assert test_X.empty
    
    def test_missing_risk_column(self):
        """Test error handling when Risk column is missing"""
        trainset = pd.DataFrame({'LoanAmount': [1000, 2000]})
        testset = pd.DataFrame({'Id': [1, 2], 'LoanAmount': [1200, 1800]})
        
        with pytest.raises(KeyError):
            prepare_features_and_target(trainset, testset)
    
    def test_missing_id_column(self, sample_datasets):
        """Test error handling when Id column is missing from test set"""
        trainset, _ = sample_datasets
        testset_no_id = pd.DataFrame({
            'LoanAmount': [1200, 1800],
            'Age': [28, 38]
        })
        
        with pytest.raises(KeyError):
            prepare_features_and_target(trainset, testset_no_id)


# Integration tests
class TestDataLoaderIntegration:
    """Integration tests for data_loader module"""
    
    @pytest.fixture
    def complete_sample_data(self):
        """Create complete sample data with realistic German credit features"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # More realistic training data
            train_data = pd.DataFrame({
                'Risk': ['Risk', 'No Risk', 'Risk', 'No Risk', 'Risk', 'No Risk'],
                'LoanAmount': [1000, 2000, 1500, 3000, 2500, 1800],
                'LoanDuration': [12, 24, 18, 36, 30, 20],
                'Age': [25, 35, 45, 55, 40, 30],
                'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent', 'Good', 'Bad'],
                'Purpose': ['Car', 'House', 'Education', 'Business', 'Car', 'House'],
                'Employment': ['Skilled', 'Unskilled', 'Skilled', 'Management', 'Skilled', 'Unskilled'],
                'PersonalStatus': ['Single', 'Married', 'Single', 'Married', 'Divorced', 'Single'],
                'Housing': ['Own', 'Rent', 'Own', 'Own', 'Rent', 'Own']
            })
            
            test_data = pd.DataFrame({
                'Id': [1, 2, 3, 4],
                'LoanAmount': [1200, 1800, 2500, 3500],
                'LoanDuration': [15, 20, 30, 40],
                'Age': [28, 38, 48, 58],
                'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent'],
                'Purpose': ['Car', 'House', 'Education', 'Business'],
                'Employment': ['Skilled', 'Unskilled', 'Management', 'Skilled'],
                'PersonalStatus': ['Single', 'Married', 'Single', 'Divorced'],
                'Housing': ['Own', 'Rent', 'Own', 'Own']
            })
            
            train_path = Path(temp_dir) / 'german_credit_train.csv'
            test_path = Path(temp_dir) / 'german_credit_test.csv'
            
            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            yield temp_dir
    
    def test_full_pipeline(self, complete_sample_data):
        """Test the complete data loading and preparation pipeline"""
        # Load data
        trainset, testset = load_german_credit_data(complete_sample_data)
        
        # Prepare features and target
        X, y, test_X = prepare_features_and_target(trainset, testset)
        
        # Comprehensive checks
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(test_X, pd.DataFrame)
        
        # Check data integrity
        assert len(X) == len(y) == 6
        assert len(test_X) == 4
        
        # Check that all categorical columns are properly converted
        categorical_cols = X.select_dtypes(include='category').columns
        assert len(categorical_cols) > 0
        
        # Check that target is binary
        assert set(y.unique()).issubset({0, 1})
        
        # Check column consistency
        assert list(X.columns) == list(test_X.columns)