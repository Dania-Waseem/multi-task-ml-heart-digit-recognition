# Heart Disease Prediction & MNIST Digit Recognition System

---

## Project Overview
This project combines two machine learning systems in a single interactive application.

The first system predicts heart disease risk using a trained Random Forest model based on 13 clinical attributes such as age, blood pressure, cholesterol, and heart rate. The model outputs a prediction along with confidence and highlights the most influential medical factors behind the decision.

The second system uses a lightweight Convolutional Neural Network trained on the MNIST dataset to classify handwritten digits from 0 to 9. This simulates real-world digit recognition tasks such as reading handwritten medical forms in healthcare environments.

Together, these components demonstrate both tabular machine learning and deep learning applications within a single pipeline.

---

## How to Run

### 1. Create Virtual Environment
python -m venv venv  
venv\Scripts\activate  

### 2. Install Dependencies
pip install -r requirements.txt  

### 3. Run Application
python -m streamlit run app.py  

Open in browser

---

## Features

### Input Interface
- 13 medical input fields for patient data
- Pre-filled sample values for quick testing
- Clean dropdowns and numeric inputs
- Single Predict button for inference

---

### Output Panel
After prediction, the system displays:
- Final prediction (Disease or No Disease)
- Confidence score in percentage form
- Top 3 most important features influencing the decision
- Simple explanation in plain language for interpretation

---

## Models Used

### Heart Disease Model
- Random Forest Classifier
- Trained on UCI Heart Disease dataset
- Preprocessing includes:
  - One-hot encoding for categorical variables
  - Standard scaling for numerical variables
  - Stratified train-test split

---

### MNIST Digit Model
- Convolutional Neural Network (CNN)
- Trained on MNIST handwritten digit dataset
- Architecture includes convolution layers, pooling layers, and dense layers
- Used for digit classification from 0 to 9

---

## Requirements
- streamlit  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- joblib  
- tensorflow  

---

## Notes
- Ensure all .pkl files are in the same directory as app.py
- Always run using Streamlit command
- If port is busy, use an alternate port option

_ _
