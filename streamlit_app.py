import sys
import os
import pandas as pd
import streamlit as st
import joblib

# == Verify Python environment ==
st.write("Python executable:", sys.executable)
st.write("Python version:", sys.version)

# == Load Model ==
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
    st.stop()  
else:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")

# == Prediction Function ==
def get_prediction(data: pd.DataFrame):
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", 
                    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
 
    
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# == Streamlit App ==
st.title("Bank Customer Churn Prediction")

# Layout with two columns
left, right = st.columns(2, gap="medium")

# -- Numeric Inputs
left.subheader("Numeric Features")
credit_score = left.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
age = left.slider("Age", min_value=18, max_value=100, value=35)
tenure = left.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = left.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
num_of_products = left.slider("Number of Products", min_value=1, max_value=4, value=1)
estimated_salary = left.number_input("Estimated Salary", min_value=1000.0, max_value=200000.0, value=50000.0, step=1000.0)

# -- Categorical Inputs
right.subheader("Categorical Features")
gender = right.selectbox("Gender", options=["Male", "Female"])
geography = right.selectbox("Geography", options=["France", "Germany", "Spain"])
has_cr_card = right.selectbox("Has Credit Card", options=[0, 1])
is_active_member = right.selectbox("Is Active Member", options=[0, 1])

# -- Prepare DataFrame for Model
data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

st.subheader("Input Data")
st.dataframe(data, use_container_width=True)

# -- Prediction Button
if st.button("Predict", use_container_width=True):
    pred, pred_proba = get_prediction(data)
    label_map = {0: "Stayed", 1: "Exited"}
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    st.success(f"Prediction: {label_pred} with probability {label_proba:.2%}")