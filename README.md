# Diabetes Risk Prediction System

## Overview
This project implements a machine learning-based system for predicting diabetes risk using health metrics. It includes both a robust machine learning pipeline and a user-friendly web interface built with Streamlit.

## Features
- Advanced preprocessing pipeline with automated feature selection
- Balanced dataset handling using downsampling techniques
- K-Nearest Neighbors classification with optimized hyperparameters
- Interactive web interface for real-time predictions
- Comprehensive error handling and logging
- Visualization of risk assessment using gauge charts
- Detailed health recommendations based on prediction results

## Directory Structure
```
diabetes-prediction/
├── data/
│   └── Diabetes_classification.xlsx
├── models/
│   └── diabetes_model.joblib
├── src/
│   ├── diabetes_classifier.py
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model using your own dataset:

```python
from diabetes_classifier import DiabetesClassifier

# Initialize and train the classifier
classifier = DiabetesClassifier()
metrics = classifier.train(your_dataframe)

# Save the trained model
classifier.save_model()
```

### Running the Web Application
To start the Streamlit web interface:

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

## Model Details
The system uses a K-Nearest Neighbors classifier with the following features:
- Glucose Level
- BMI
- Blood Pressure (Systolic and Diastolic)
- Cholesterol Metrics
- Waist/Hip Ratio
- Weight

The model is optimized using GridSearchCV with cross-validation to find the best hyperparameters.

## Web Interface
The Streamlit interface provides:
- Easy input of health metrics
- Real-time risk prediction
- Visual representation of risk level
- Personalized health recommendations
- Comprehensive error handling

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- plotly
- joblib

See `requirements.txt` for complete list of dependencies.

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: [Include source of the diabetes dataset]
- Streamlit for the amazing web framework
- scikit-learn for machine learning tools

## Contact
Your Name - [@yourtwitter](https://twitter.com/yourtwitter)
Project Link: [https://github.com/mr-imperium/diabetes-prediction](https://github.com/mr-imperium/diabetes-prediction)