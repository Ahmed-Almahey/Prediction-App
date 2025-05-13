import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Customer Churn Predictor")
st.write("This app uses a saved Naive Bayes model to predict customer churn.")

churn_df = pd.read_excel("churn_dataset.xlsx")
st.subheader("📊 Show Dataset Preview")
st.write(churn_df.head())

st.sidebar.header("Input Customer Info")

age = st.sidebar.slider("Customer Age",
                        churn_df['Age'].min(),
                        churn_df['Age'].max(),
                        int(churn_df['Age'].mean()))

tenure = st.sidebar.slider("Customer Tenure",
                           churn_df['Tenure'].min(),
                           churn_df['Tenure'].max(),
                           int(churn_df['Tenure'].mean()))

gender_text = st.sidebar.selectbox("Customer Gender", ["Female", "Male"])
gender = 0 if gender_text == "Female" else 1  # Map to numeric if model needs it

# Input data (update based on your model's expected features)
input_data = np.array([[age, tenure, gender]])

# Load model
model = joblib.load("NaiveBayesModel.pkl")

# Predict
prediction = model.predict(input_data)[0]
pred_proba = model.predict_proba(input_data)[0]

# Prediction result mapping
result_meaning = {
    1: "Yes, will churn.",
    0: "No, won't churn."
}

# Show Prediction
st.image("https://emojikitchen.com/emoji/noto-animated/1f52e/emoji.gif", width=80)
st.subheader("🌟 Prediction Result")
st.write(f"**Predicted Result:** {result_meaning[prediction]}")

# Show Prediction Probabilities
st.write("📈 Prediction Probabilities")
st.write(f"No Churn: {pred_proba[0]:.2%}")
st.write(f"Churn: {pred_proba[1]:.2%}")
st.markdown("---")
st.caption("Designed with 💡 and ❤️ by Ahmed Almahey")

