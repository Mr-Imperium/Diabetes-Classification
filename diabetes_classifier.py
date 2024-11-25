import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
import logging
import pickle
from pathlib import Path

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DiabetesClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.best_features = None
        self.feature_ranges = None
        self.X_train = None
        self.y_train = None
        self.feature_descriptions = {
            "Cholesterol": "Total cholesterol (mg/dL)",
            "Glucose": "Fasting blood sugar (mg/dL)",
            "HDL Chol": "HDL (good) cholesterol (mg/dL)",
            "Chol/HDL ratio": "Ratio of total cholesterol to HDL (Desirable < 5)",
            "Age": "Age in years",
            "Height": "Height in inches",
            "Weight": "Weight in pounds",
            "BMI": "Body Mass Index (703 × weight/height²)",
            "Systolic BP": "Systolic Blood Pressure (mmHg)",
            "Diastolic BP": "Diastolic Blood Pressure (mmHg)",
            "Waist": "Waist measurement in inches",
            "Hip": "Hip measurement in inches",
            "Waist/hip ratio": "Waist to hip ratio (heart disease risk factor)"
        }
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input dataframe by selecting relevant features and handling missing values."""
        try:
            # Log initial dataframe shape
            logging.info(f"Initial dataframe shape: {df.shape}")
            logging.info(f"Initial columns: {df.columns.tolist()}")
            
            # Select relevant features
            selected_features = [
                "Cholesterol", "Glucose", "HDL Chol", "Chol/HDL ratio", 
                "Age", "BMI", "Systolic BP", "Diastolic BP", 
                "Waist/hip ratio", "Weight", "Height"
            ]
            
            # Log missing columns
            missing_cols = [col for col in selected_features + ["Diabetes"] if col not in df.columns]
            if missing_cols:
                logging.error(f"Missing required columns: {', '.join(missing_cols)}")
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Create reduced dataframe
            df_reduced = df[["Diabetes"] + selected_features].copy()
            logging.info(f"Shape after selecting features: {df_reduced.shape}")
            
            # Log unique values in Diabetes column before cleaning
            logging.info(f"Unique values in Diabetes column before cleaning: {df_reduced['Diabetes'].unique()}")
            
            # Clean up Diabetes column with updated mapping
            df_reduced['Diabetes'] = df_reduced['Diabetes'].str.strip()
            df_reduced['Diabetes'] = df_reduced['Diabetes'].map({
                'Diabetes': 1,
                'No diabetes': 0,
                'Yes': 1, 'Y': 1, '1': 1, True: 1, 
                'No': 0, 'N': 0, '0': 0, False: 0
            })
            
            # Log unique values after mapping
            logging.info(f"Unique values in Diabetes column after mapping: {df_reduced['Diabetes'].unique()}")
            
            # Remove rows where Diabetes status is unknown
            initial_rows = len(df_reduced)
            df_reduced = df_reduced.dropna(subset=['Diabetes'])
            rows_removed = initial_rows - len(df_reduced)
            logging.info(f"Rows removed due to missing Diabetes status: {rows_removed}")
            logging.info(f"Shape after removing missing Diabetes values: {df_reduced.shape}")
            
            # Log missing values in feature columns
            for col in selected_features:
                missing_count = df_reduced[col].isna().sum()
                if missing_count > 0:
                    logging.info(f"Missing values in {col}: {missing_count}")
                if df_reduced[col].isna().any():
                    median_value = df_reduced[col].median()
                    df_reduced[col] = df_reduced[col].fillna(median_value)
            
            # Store feature ranges for validation with descriptions
            self.feature_ranges = {
                feature: {
                    'min': df_reduced[feature].min(),
                    'max': df_reduced[feature].max(),
                    'mean': df_reduced[feature].mean(),
                    'std': df_reduced[feature].std(),
                    'description': self.feature_descriptions.get(feature, '')
                }
                for feature in selected_features
            }
            
            self.feature_names = selected_features
            logging.info(f"Final preprocessed dataframe shape: {df_reduced.shape}")
            
            # Verify we have data
            if len(df_reduced) == 0:
                logging.error("Preprocessed dataframe is empty!")
                raise ValueError("Preprocessed dataframe is empty! Check the data cleaning steps.")
            
            return df_reduced
        
        except Exception as e:
            logging.error(f"Error in preprocessing data: {str(e)}")
            raise
            
    def validate_input(self, features: Dict[str, float]) -> Tuple[bool, list]:
        """Validate input features against training data ranges."""
        if not self.feature_ranges:
            raise ValueError("Model not trained - feature ranges not available")
            
        warnings = []
        is_valid = True
        
        for feature, value in features.items():
            if feature in self.feature_ranges:
                range_info = self.feature_ranges[feature]
                # Check if value is within 3 standard deviations of the mean
                lower_bound = range_info['mean'] - 3 * range_info['std']
                upper_bound = range_info['mean'] + 3 * range_info['std']
                
                if value < lower_bound or value > upper_bound:
                    warnings.append(f"{feature}: {value} is outside normal range "
                                 f"({lower_bound:.1f} - {upper_bound:.1f})")
                    is_valid = False
                    
                # Additional validation for specific metrics
                if feature == "Chol/HDL ratio" and value < 5:
                    warnings.append("Chol/HDL ratio is in the desirable range (< 5)")
                    
        return is_valid, warnings

    def calculate_bmi(self, weight: float, height: float) -> float:
        """Calculate BMI using the formula: 703 × weight(lbs)/height(inches)²"""
        return 703 * weight / (height ** 2)

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the model using the provided dataset."""
        try:
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Split features and target
            X = df_processed.drop('Diabetes', axis=1)
            y = df_processed['Diabetes']
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Feature selection
            selector = SelectKBest(score_func=f_classif, k=8)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            self.best_features = X_train.columns[selected_mask].tolist()
            
            # Store training data for later use
            self.X_train = X_train_selected
            self.y_train = y_train
            
            # Define parameter grid for GridSearchCV
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            
            # Initialize and train model with GridSearchCV
            self.model = GridSearchCV(
                KNeighborsClassifier(), 
                param_grid, 
                cv=5, 
                scoring='accuracy'
            )
            self.model.fit(X_train_selected, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test_selected)
            
            # Calculate metrics
            results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'best_params': self.model.best_params_
            }
            
            logging.info("Model trained successfully")
            logging.info(f"Best parameters: {results['best_params']}")
            logging.info(f"Accuracy: {results['accuracy']:.2f}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in training model: {str(e)}")
            raise

    def predict(self, input_data):
        # Ensure input is a DataFrame with correct columns
        if isinstance(input_data, dict):
            # Create DataFrame with selected features in the correct order
            input_df = pd.DataFrame([input_data])[self.selected_features]
        elif isinstance(input_data, pd.DataFrame):
            # Ensure correct column order
            input_df = input_data[self.selected_features]
        else:
            # Convert to DataFrame assuming input is an array-like
            input_df = pd.DataFrame(input_data, columns=self.selected_features)
        
        # Prepare features for prediction
        try:
            # Ensure input is numeric and matches expected feature types
            feature_array = input_df.astype(float)
            
            # Transform using the scaler
            scaled_features = self.scaler.transform(feature_array)
            
            # Predict probability
            probability = self.model.predict_proba(scaled_features)[:, 1]
            prediction = (probability > 0.5).astype(int)
            
            return prediction[0], probability[0]
        
        except Exception as e:
            # Add detailed error logging
            st.error(f"Prediction error: {e}")
            st.write("Input data:", input_df)
            st.write("Input data types:", input_df.dtypes)
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.model or not self.best_features:
            raise ValueError("Model not trained yet!")
            
        # For KNN, we'll use the feature selection scores as importance
        selector = SelectKBest(score_func=f_classif, k=len(self.best_features))
        X = pd.DataFrame(self.scaler.inverse_transform(self.X_train), columns=self.best_features)
        selector.fit(X, self.y_train)
        
        importance_dict = dict(zip(self.best_features, selector.scores_))
        return importance_dict

    def get_feature_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get the acceptable ranges for each feature."""
        if not self.feature_ranges:
            raise ValueError("Model not trained - feature ranges not available")
        return self.feature_ranges

    def save_model(self, path: str = 'diabetes_model.pkl'):
        """Save the trained model and preprocessors."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'best_features': self.best_features,
                'feature_ranges': self.feature_ranges,
                'X_train': self.X_train,
                'y_train': self.y_train
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            logging.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str = 'diabetes_model.pkl'):
        """Load a trained model and preprocessors."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.best_features = model_data['best_features']
            self.feature_ranges = model_data['feature_ranges']
            self.X_train = model_data['X_train']
            self.y_train = model_data['y_train']
            logging.info(f"Model loaded successfully from {path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
