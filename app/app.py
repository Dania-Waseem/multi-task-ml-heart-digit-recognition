import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load artifacts
model     = joblib.load("best_rf.pkl")
scaler    = joblib.load("scaler.pkl")
cat_cols  = joblib.load("cat_cols.pkl")
num_cols  = joblib.load("num_cols.pkl")
X_train   = joblib.load("X_train.pkl")

st.title("CardioAI — Heart Disease Predictor")

st.header("Patient Input Form")

# Input fields (pre-filled with one real test patient)
age      = st.number_input("Age (20–80)", 20, 80, 54)
sex      = st.selectbox("Sex (1=Male, 0=Female)", [1, 0], index=0)
cp       = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3], index=2)
trestbps = st.number_input("Resting BP (90–200)", 90, 200, 130)
chol     = st.number_input("Cholesterol (100–600)", 100, 600, 245)
fbs      = st.selectbox("Fasting Blood Sugar >120", [0, 1], index=0)
restecg  = st.selectbox("Resting ECG (0–2)", [0, 1, 2], index=1)
thalach  = st.number_input("Max Heart Rate (70–210)", 70, 210, 150)
exang    = st.selectbox("Exercise Angina", [0, 1], index=0)
oldpeak  = st.number_input("ST Depression (0.0–6.0)", 0.0, 6.0, 1.5)
slope    = st.selectbox("ST Slope (0–2)", [0, 1, 2], index=1)
ca       = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3], index=0)
thal     = st.selectbox("Thal (1–3)", [1, 2, 3], index=2)

if st.button("Predict"):

    # Create input
    raw = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope,
        "ca": ca, "thal": thal
    }])

    # Encode + align
    enc = pd.get_dummies(raw, columns=cat_cols)
    enc = enc.reindex(columns=X_train.columns, fill_value=0)
    enc[num_cols] = scaler.transform(enc[num_cols])

    # Prediction
    prob = model.predict_proba(enc)[0][1]
    pred = int(prob >= 0.5)

    # Result display (E2)
    if pred == 1:
        st.error(f"Disease Present — Confidence: {prob*100:.1f}%")
    else:
        st.success(f"No Disease — Confidence: {(1-prob)*100:.1f}%")

    # Top features
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top3 = importances.sort_values(ascending=False).head(3)

    fig, ax = plt.subplots()
    top3.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("Top 3 Features Driving Prediction")
    st.pyplot(fig)

    # Explanation text
    st.write(
        f"This prediction is mainly influenced by {', '.join(top3.index)}. "
        f"These features strongly affect heart disease risk based on the trained model."
    )