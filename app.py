import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from diabetes_classifier import DiabetesClassifier
import plotly.express as px
import plotly.graph_objects as go

# Initialize the classifier
@st.cache_resource
def load_model():
    classifier = DiabetesClassifier()
    try:
        # Add print statement to debug file path
        import os
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        print(f"Attempting to load model from: {os.path.join(current_dir, 'diabetes_model.pkl')}")
        
        classifier.load_model('diabetes_model.pkl')
        return classifier
    except FileNotFoundError as e:
        # More specific error message
        st.error(f"Model file not found: {e}")
        return None
    except Exception as e:
        # Catch and display any other exceptions
        st.error(f"Error loading model: {e}")
        return None

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
    
    st.title("Diabetes Risk Assessment Tool")
    st.write("""
    This application uses machine learning to assess diabetes risk based on various health metrics.
    Please enter your health information below for an assessment.
    """)
    
    # Load the model
    classifier = load_model()
    if not classifier:
        return

    # Define the exact feature order
    REQUIRED_FEATURES = [
        "Cholesterol", "Glucose", "HDL Chol", "Chol/HDL ratio",
        "Age", "BMI", "Systolic BP", "Diastolic BP",
        "Waist/hip ratio", "Weight", "Height"
    ]
        
    # Get feature ranges and descriptions
    ranges = classifier.get_feature_ranges()
    
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store user inputs
    user_inputs = {}
    
    # Create input fields with validation
    with col1:
        st.subheader("Basic Information")
        user_inputs["Age"] = st.number_input(
            "Age (years)", 
            min_value=int(ranges["Age"]["min"]),
            max_value=int(ranges["Age"]["max"]),
            value=int(ranges["Age"]["mean"])
        )
        
        user_inputs["Height"] = st.number_input(
            "Height (inches)",
            min_value=float(ranges["Height"]["min"]),
            max_value=float(ranges["Height"]["max"]),
            value=float(ranges["Height"]["mean"]),
            format="%.1f"
        )
        
        user_inputs["Weight"] = st.number_input(
            "Weight (pounds)",
            min_value=float(ranges["Weight"]["min"]),
            max_value=float(ranges["Weight"]["max"]),
            value=float(ranges["Weight"]["mean"]),
            format="%.1f"
        )
        
        # Calculate BMI automatically
        user_inputs["BMI"] = classifier.calculate_bmi(user_inputs["Weight"], user_inputs["Height"])
        st.info(f"Calculated BMI: {user_inputs['BMI']:.1f}")

    with col2:
        st.subheader("Blood Pressure & Body Measurements")
        user_inputs["Systolic BP"] = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=int(ranges["Systolic BP"]["min"]),
            max_value=int(ranges["Systolic BP"]["max"]),
            value=int(ranges["Systolic BP"]["mean"])
        )
        
        user_inputs["Diastolic BP"] = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=int(ranges["Diastolic BP"]["min"]),
            max_value=int(ranges["Diastolic BP"]["max"]),
            value=int(ranges["Diastolic BP"]["mean"])
        )
        
        # Use fallback values if 'Waist' is not in ranges
        waist_min = float(ranges.get("Waist", {"min": 0})["min"])
        waist_max = float(ranges.get("Waist", {"max": 100})["max"])
        waist_mean = float(ranges.get("Waist", {"mean": 35})["mean"])
        
        waist = st.number_input(
            "Waist (inches)",
            min_value=waist_min,
            max_value=waist_max,
            value=waist_mean,
            format="%.1f"
        )
        
        hip_min = float(ranges.get("Hip", {"min": 0})["min"])
        hip_max = float(ranges.get("Hip", {"max": 150})["max"])
        hip_mean = float(ranges.get("Hip", {"mean": 40})["mean"])
        
        hip = st.number_input(
            "Hip (inches)",
            min_value=hip_min,
            max_value=hip_max,
            value=hip_mean,
            format="%.1f"
        )
        # Calculate Waist/hip ratio automatically
        user_inputs["Waist/hip ratio"] = waist / hip if hip != 0 else 0
        st.info(f"Calculated Waist/Hip Ratio: {user_inputs['Waist/hip ratio']:.2f}")

    with col3:
        st.subheader("Cholesterol Metrics")
        user_inputs["Cholesterol"] = st.number_input(
            "Total Cholesterol (mg/dL)",
            min_value=int(ranges["Cholesterol"]["min"]),
            max_value=int(ranges["Cholesterol"]["max"]),
            value=int(ranges["Cholesterol"]["mean"])
        )
        
        user_inputs["HDL Chol"] = st.number_input(
            "HDL Cholesterol (mg/dL)",
            min_value=int(ranges["HDL Chol"]["min"]),
            max_value=int(ranges["HDL Chol"]["max"]),
            value=int(ranges["HDL Chol"]["mean"])
        )
        
        # Calculate Chol/HDL ratio automatically
        user_inputs["Chol/HDL ratio"] = user_inputs["Cholesterol"] / user_inputs["HDL Chol"] if user_inputs["HDL Chol"] != 0 else 0
        st.info(f"Calculated Cholesterol/HDL Ratio: {user_inputs['Chol/HDL ratio']:.2f}")
        
        user_inputs["Glucose"] = st.number_input(
            "Fasting Blood Glucose (mg/dL)",
            min_value=int(ranges["Glucose"]["min"]),
            max_value=int(ranges["Glucose"]["max"]),
            value=int(ranges["Glucose"]["mean"])
        )

    # Add a predict button
    if st.button("Assess Diabetes Risk"):
        # Validate inputs
        is_valid, warnings = classifier.validate_input(user_inputs)
        
        # Display warnings if any
        if warnings:
            st.warning("Warnings:")
            for warning in warnings:
                st.write(f"- {warning}")

       
        # Convert user inputs to a DataFrame with selected features
        prediction_input = pd.DataFrame([user_inputs])[SELECTED_FEATURES]

        # Make prediction
        prediction, probability = classifier.predict(prediction_input)
        
        # Display results
        st.header("Assessment Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Assessment")
            fig = create_gauge_chart(probability, "Diabetes Risk Score")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Key Factors")
            # Get feature importance
            importance = classifier.get_feature_importance()
            importance_df = pd.DataFrame({
                'Feature': importance.keys(),
                'Importance': importance.values()
            })
            importance_df = importance_df.sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Feature Importance')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display interpretation
        st.subheader("Interpretation")
        risk_level = "high" if probability > 0.7 else "moderate" if probability > 0.3 else "low"
        st.write(f"""
        Based on the provided health metrics, our model indicates a **{risk_level} risk** of diabetes, 
        with a probability of {probability:.1%}. This assessment is based on a machine learning model 
        trained on historical health data.
        
        Please note that this is not a medical diagnosis. Consult with a healthcare provider for proper 
        medical advice and diagnosis.
        """)
        
        # Recommendations
        st.subheader("General Health Recommendations")
        st.write("""
        Regardless of risk level, consider these general health recommendations:
        - Maintain a balanced, healthy diet
        - Regular physical activity (at least 150 minutes per week)
        - Regular health check-ups with your healthcare provider
        - Monitor blood glucose levels as recommended by your doctor
        - Maintain a healthy weight
        """)

if __name__ == "__main__":
    main()
