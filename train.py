import pandas as pd
from diabetes_classifier import DiabetesClassifier

def main():
    # Read the Excel file
    print("Reading data...")
    df = pd.read_excel('Diabetes_Classification.xlsx')
    
    # Initialize the classifier
    print("Initializing classifier...")
    classifier = DiabetesClassifier()
    
    # Train the model
    print("Initial data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample of first few rows:")
    print(df.head())
    print("Training model...")
    results = classifier.train(df)
    
    # Print results
    print("\nTraining Results:")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print("\nBest Parameters:", results['best_params'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save the model
    print("\nSaving model...")
    classifier.save_model('diabetes_model.pkl')
    print("Model saved as 'diabetes_model.pkl'")

if __name__ == "__main__":
    main()