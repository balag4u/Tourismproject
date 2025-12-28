# for web application UI
import streamlit as st

# for data manipulation
import pandas as pd

# for model loading from Hugging Face
from huggingface_hub import hf_hub_download

# for model deserialization
import joblib


# Download the trained tourism model from Hugging Face
model_path = hf_hub_download(
    repo_id="nsa9/tourism-best-model",
    filename="best_tourism_model.joblib"
)

# Load the model
model = joblib.load(model_path)


# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write(
    "This application predicts whether a customer is likely to purchase a travel package "
    "based on their demographic details and interaction history."
)
st.write("Please enter the customer information below.")


# Collect user input
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5], index=3)
NumberOfTrips = st.number_input("Number of Trips Taken Earlier", min_value=0, max_value=50, value=2)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)

TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])


# Prepare input data for prediction
input_data = pd.DataFrame([{
    "Age": Age,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
}])


# Set classification threshold
classification_threshold = 0.45


# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    result = (
        "likely to purchase the travel package"
        if prediction == 1
        else "unlikely to purchase the travel package"
    )

    st.write(f"Based on the information provided, the customer is {result}.")
