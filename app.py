from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Initialize scalers
scalers = {}

def train_slr():
    """
    Train Simple Linear Regression model for loan amount prediction
    Features: CIBIL Score
    Target: Loan Amount
    """
    try:
        df = pd.read_csv("datasets/slr_data.csv")
        X = df[['CIBIL_Score']]
        y = df['Loan_Amount']
        model = LinearRegression().fit(X, y)
        joblib.dump(model, "models/slr_model.pkl")
        return True
    except Exception as e:
        print(f"Error training SLR model: {str(e)}")
        return False

def train_mlr():
    """
    Train Multiple Linear Regression model for interest rate prediction
    Features: CIBIL Score, Annual Income, Age
    Target: Interest Rate
    """
    try:
        df = pd.read_csv("datasets/mlr_data.csv")
        X = df[['CIBIL_Score', 'Annual_Income', 'Age']]
        y = df['Interest_Rate']
        model = LinearRegression().fit(X, y)
        joblib.dump(model, "models/mlr_model.pkl")
        return True
    except Exception as e:
        print(f"Error training MLR model: {str(e)}")
        return False

def train_logistic():
    """
    Train Logistic Regression model for loan approval prediction
    Features: CIBIL Score
    Target: Approval Status (0/1)
    """
    try:
        df = pd.read_csv("datasets/logistic_data.csv")
        X = df[['CIBIL_Score']]
        y = df['Approval_Status']
        model = LogisticRegression().fit(X, y)
        joblib.dump(model, "models/logistic_model.pkl")
        return True
    except Exception as e:
        print(f"Error training Logistic model: {str(e)}")
        return False

def train_polynomial():
    """
    Train Polynomial Regression model for advanced interest rate prediction
    Features: Loan Amount
    Target: Interest Rate
    """
    try:
        df = pd.read_csv("datasets/poly_data.csv")
        X = df[['Loan_Amount']]
        y = df['Interest_Rate']
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        joblib.dump((model, poly), "models/polynomial_model.pkl")
        return True
    except Exception as e:
        print(f"Error training Polynomial model: {str(e)}")
        return False

def train_knn():
    """
    Train K-Nearest Neighbors model for default risk prediction
    Features: CIBIL Score, Monthly Income
    Target: Default Risk (0: Low Risk, 1: High Risk)
    """
    try:
        df = pd.read_csv("datasets/knn_data.csv")
        X = df[['CIBIL_Score', 'Monthly_Income']]
        y = df['Default_Risk']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train KNN model
        model = KNeighborsClassifier(n_neighbors=3).fit(X_scaled, y)
        
        # Save both model and scaler
        joblib.dump(model, "models/knn_model.pkl")
        joblib.dump(scaler, "models/knn_scaler.pkl")
        return True
    except Exception as e:
        print(f"Error training KNN model: {str(e)}")
        return False

# Create models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Train all models
models_trained = {
    'slr': train_slr(),
    'mlr': train_mlr(),
    'logistic': train_logistic(),
    'polynomial': train_polynomial(),
    'knn': train_knn()
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_loan_amount')
def predict_loan_amount():
    return render_template('slr.html')

@app.route('/predict_interest_rate')
def predict_interest_rate():
    return render_template('mlr.html')

@app.route('/predict_approval_status')
def predict_approval_status():
    return render_template('logistic.html')

@app.route('/predict_interest_rate_poly')
def predict_interest_rate_poly():
    return render_template('polynomial.html')

@app.route('/predict_default_risk')
def predict_default_risk():
    return render_template('knn.html')

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        if not models_trained.get(model_name, False):
            return jsonify({'error': f'Model {model_name} is not properly trained'})

        data = request.json.get('features')
        if not data:
            return jsonify({'error': 'No features provided'})

        model_path = f"models/{model_name}_model.pkl"
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'})
        
        if model_name == 'knn':
            # Load KNN model and scaler
            model = joblib.load(model_path)
            scaler = joblib.load("models/knn_scaler.pkl")
            
            # Scale the input features
            features_scaled = scaler.transform([data])
            prediction = model.predict(features_scaled)
            
            # Return human-readable prediction
            return jsonify({'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk'})
        
        elif model_name == 'polynomial':
            model_tuple = joblib.load(model_path)
            model, poly = model_tuple
            features = poly.transform([[data[0]]])
            prediction = model.predict(features)
            return jsonify({'prediction': f"{prediction[0]:.2f}%"})
        
        else:
            model = joblib.load(model_path)
            prediction = model.predict([data])
            
            # Format prediction based on model type
            if model_name == 'slr':
                return jsonify({'prediction': f"₹{prediction[0]:,.2f}"})
            elif model_name == 'mlr':
                return jsonify({'prediction': f"{prediction[0]:.2f}%"})
            elif model_name == 'logistic':
                return jsonify({'prediction': 'Approved' if prediction[0] == 1 else 'Not Approved'})
            
            return jsonify({'prediction': prediction.tolist()})
            
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 