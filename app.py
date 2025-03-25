from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load datasets and train models
models = {}

def train_slr():
    df = pd.read_csv("datasets/slr_data.csv")
    X = df[['CIBIL_Score']]
    y = df['Approval_Probability']
    model = LinearRegression().fit(X, y)
    joblib.dump(model, "models/slr_model.pkl")

def train_mlr():
    df = pd.read_csv("datasets/mlr_data.csv")
    X = df[['CIBIL_Score', 'Annual_Income', 'Age']]
    y = df['Loan_Amount']
    model = LinearRegression().fit(X, y)
    joblib.dump(model, "models/mlr_model.pkl")

def train_logistic():
    df = pd.read_csv("datasets/logistic_data.csv")
    X = df[['CIBIL_Score']]
    y = df['Approval_Status']
    model = LogisticRegression().fit(X, y)
    joblib.dump(model, "models/logistic_model.pkl")

def train_polynomial():
    df = pd.read_csv("datasets/poly_data.csv")
    X = df[['Loan_Amount']]
    y = df['Interest_Rate']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        
    joblib.dump((model, poly), "models/polynomial_model.pkl")

def train_knn():
    df = pd.read_csv("datasets/knn_data.csv")
    X = df[['CIBIL_Score', 'Monthly_Income']]
    y = df['Default_Risk']
    model = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    joblib.dump(model, "models/knn_model.pkl")

# Train all models
train_slr()
train_mlr()
train_logistic()
train_polynomial()
train_knn()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/slr')
def slr():
    return render_template('slr.html')

@app.route('/mlr')
def mlr():
    return render_template('mlr.html')

@app.route('/logistic')
def logistic():
    return render_template('logistic.html')

@app.route('/polynomial')
def polynomial():
    return render_template('polynomial.html')

@app.route('/knn')
def knn():
    return render_template('knn.html')

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        data = request.json['features']
        model_path = f"models/{model_name}_model.pkl"
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'})
        
        if model_name == 'polynomial':
            model_tuple = joblib.load(model_path)
            model, poly = model_tuple
            features = poly.transform([[data[0]]])
            prediction = model.predict(features)
        else:
            model = joblib.load(model_path)
            prediction = model.predict([data])
        
        # Format prediction based on model type
        if model_name == 'slr':
            prediction = float(prediction[0])
            return jsonify({'prediction': f"{prediction:.2%}"})
        elif model_name == 'mlr':
            prediction = float(prediction[0])
            return jsonify({'prediction': f"₹{prediction:,.2f}"})
        elif model_name == 'logistic':
            prediction = int(prediction[0])
            return jsonify({'prediction': 'Approved' if prediction == 1 else 'Not Approved'})
        elif model_name == 'polynomial':
            prediction = float(prediction[0])
            return jsonify({'prediction': f"{prediction:.2f}%"})
        elif model_name == 'knn':
            prediction = int(prediction[0])
            return jsonify({'prediction': 'High Risk' if prediction == 1 else 'Low Risk'})
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 