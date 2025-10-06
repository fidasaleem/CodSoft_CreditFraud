import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# Load saved LightGBM model
model = joblib.load('lightgbm_fraud_model.pkl')  
scaler_std = StandardScaler() 
# Streamlit App
st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether it is fraudulent or genuine.")

# Input fields
Time = st.number_input("Time", min_value=0.0, value=0.0)
Amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0)

# V1 to V28 inputs
v_values = {}
for i in range(1, 29):
    v_values[f'V{i}'] = st.number_input(f"V{i}", value=0.0)

# Predict button
if st.button("Predict Fraud"):
    Hour = (Time // 3600) % 24
    Day = (Time // (3600*24)) + 1
    Amount_log = np.log1p(Amount)
    Amount_log_scaled = Amount_log
    # Create input DataFrame
    input_dict = {
        'Time': Time,
        'Amount': Amount,
        'Hour': Hour,
        'Day': Day,
        'Amount_log': Amount_log,
        'Amount_log_scaled': Amount_log_scaled,
        **v_values
    }
    input_df = pd.DataFrame([input_dict])
    
    # Ensure all columns are present in the same order as training
    feature_names = model.booster_.feature_name()  # LightGBM feature order
    missing_cols = [c for c in feature_names if c not in input_df.columns]
    for c in missing_cols:
        input_df[c] = 0  # default for missing columns
    
    input_df = input_df[feature_names] 
    
    # Predict probability
    fraud_proba = model.predict_proba(input_df)[:, 1][0]
    
    # Threshold from your tuning
    threshold = 0.41
    prediction = "Fraud" if fraud_proba >= threshold else "Genuine"
    
    # Display result
    st.subheader("Prediction Result")
    st.write(f"Transaction is **{prediction}**")
    st.write(f"Fraud Probability: **{fraud_proba:.2f}**")
