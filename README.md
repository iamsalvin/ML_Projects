# Machine Learning Models Dashboard

This project is a web application that provides a user-friendly interface for interacting with various machine learning models. The application includes implementations of:

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression
- Polynomial Regression
- k-Nearest Neighbors (kNN)

## Project Structure

```
ML_Combined_Project/
│── static/                  # Static files (CSS, JS, images)
│   ├── style.css           # Styling for all pages
│── templates/              # HTML templates
│   ├── index.html         # Main dashboard
│   ├── slr.html           # Simple Linear Regression UI
│   ├── mlr.html           # Multiple Linear Regression UI
│   ├── logistic.html      # Logistic Regression UI
│   ├── polynomial.html    # Polynomial Regression UI
│   ├── knn.html           # kNN UI
│── models/                 # Saved ML models
│   ├── slr_model.pkl      # SLR Model
│   ├── mlr_model.pkl      # MLR Model
│   ├── logistic_model.pkl # Logistic Regression Model
│   ├── poly_model.pkl     # Polynomial Regression Model
│   ├── knn_model.pkl      # kNN Model
│── datasets/              # Dataset files
│   ├── slr_data.csv      # Dataset for SLR
│   ├── mlr_data.csv      # Dataset for MLR
│   ├── logistic_data.csv # Dataset for Logistic Regression
│   ├── poly_data.csv     # Dataset for Polynomial Regression
│   ├── knn_data.csv      # Dataset for kNN
│── app.py                # Flask backend
│── requirements.txt      # Dependencies
│── README.md            # This file
```

## Setup and Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your web browser and navigate to `http://localhost:5000`

## Features

- Interactive web interface for each machine learning model
- Real-time predictions
- Error handling and user feedback
- Responsive design
- Easy navigation between different models

## Models Description

### Simple Linear Regression

- Predicts continuous values using a single feature
- Suitable for linear relationships between variables

### Multiple Linear Regression

- Predicts continuous values using multiple features
- Handles complex relationships between multiple variables

### Logistic Regression

- Performs binary classification
- Outputs probability of belonging to a class

### Polynomial Regression

- Models non-linear relationships
- Uses polynomial features for better fit

### k-Nearest Neighbors

- Classification based on nearest neighbors
- Non-parametric method for pattern recognition

## Technologies Used

- Python 3.x
- Flask (Web Framework)
- scikit-learn (Machine Learning)
- Bootstrap 5 (Frontend)
- JavaScript (Frontend Interactivity)

## Contributing

Feel free to submit issues and enhancement requests!
