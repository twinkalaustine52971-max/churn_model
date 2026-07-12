import streamlit as st
import pickle
import joblib
import numpy as np

# Load compressed model
with open("churn_model_compressed.pkl", "rb") as f:
    model = joblib.load("Churn_Model_Compressed.pkl")

st.title("Customer Churn Prediction")

credit = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.selectbox("Number of Products", [1, 2, 3, 4])
card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)
gender = st.selectbox("Gender", ["Female", "Male"])
geo = st.selectbox("Geography", ["France", "Germany", "Spain"])

# One-Hot Encoding
gender = 1 if gender == "Male" else 0
geo_france = 1 if geo == "France" else 0
geo_germany = 1 if geo == "Germany" else 0
geo_spain = 1 if geo == "Spain" else 0

features = np.array([[credit, age, tenure, balance,
                      products, card, active, salary,
                      gender, geo_france, geo_germany, geo_spain]])

if st.button("Predict"):
    pred = model.predict(features)

    if pred[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")
