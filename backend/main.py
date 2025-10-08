import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI(title="Churn Prediction API")

# 2. Load the trained model
# This model is loaded once when the application starts
model = joblib.load('churn_model.joblib')

# 3. Define the input data model using Pydantic
# This ensures that the input data is valid
class CustomerData(BaseModel):
    TenureMonths: int
    SubscriptionType: str  # e.g., "Basic", "Standard", "Premium"
    MonthlyCharges: float
    SupportCalls: int

# 4. Define a root endpoint for health checks
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Churn Prediction API is running!"}

# 5. Define the prediction endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    """
    Receives customer data and predicts the probability of churn.
    """
    # Convert the input data into a pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # --- Data Preprocessing ---
    # Perform the same one-hot encoding as in the training script
    # This is a critical step to ensure consistency
    input_df['SubscriptionType_Premium'] = 1 if data.SubscriptionType == 'Premium' else 0
    input_df['SubscriptionType_Standard'] = 1 if data.SubscriptionType == 'Standard' else 0
    
    # Drop the original categorical column
    input_df = input_df.drop('SubscriptionType', axis=1)

    # Ensure the columns are in the same order as during training
    # The model is sensitive to column order
    expected_columns = ['TenureMonths', 'MonthlyCharges', 'SupportCalls', 'SubscriptionType_Premium', 'SubscriptionType_Standard']
    input_df = input_df[expected_columns]

    # --- Make Prediction ---
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Probability of churning

    # --- Return Response ---
    churn_status = "Will Churn" if prediction == 1 else "Will Not Churn"
    
    return {
        "prediction": churn_status,
        "probability_of_churn": f"{probability:.2%}"
    }