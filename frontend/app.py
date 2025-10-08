import streamlit as st
import requests
import os

# --- Configuration ---

API_URL = st.secrets.get("API_URL")

# --- Page Setup ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="👋",
    layout="centered"
)

st.title("👤 Customer Churn Prediction")
st.write("""
Enter the customer's details below to predict whether they are likely to churn.
This app communicates with a FastAPI backend to get the prediction.
""")

# --- Input Form ---
with st.form("prediction_form"):
    st.header("Enter Customer Details")
    
    # Input fields for the features
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12, step=1)
        support_calls = st.number_input("Support Calls", min_value=0, max_value=10, value=2, step=1)

    with col2:
        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=55.0, step=0.5)

    # Submit button
    submit_button = st.form_submit_button(label="Predict Churn")

# --- Prediction Logic ---
if submit_button:
    # Prepare the data in the format the API expects
    payload = {
        "TenureMonths": tenure,
        "SubscriptionType": subscription,
        "MonthlyCharges": monthly_charges,
        "SupportCalls": support_calls
    }

    try:
        # Send a POST request to the FastAPI backend
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            probability = result['probability_of_churn']
            
            if prediction == "Will Churn":
                st.error(f"Prediction: **{prediction}**")
                st.warning(f"Probability of Churn: **{probability}**")
            else:
                st.success(f"Prediction: **{prediction}**")
                st.info(f"Probability of Churn: **{probability}**")
        else:
            st.error(f"Error: Could not get a prediction. Status code: {response.status_code}")
            st.error(response.text)

    except requests.exceptions.RequestException as e:

        st.error(f"An error occurred while connecting to the API: {e}")
