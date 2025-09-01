# app.py
import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="📚 Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.markdown("Enter the details below to predict **student's performance** using ML.")

# Input fields
st.subheader("📌 Enter Student Information")

gender = st.selectbox("Gender", ["male", "female"])
parent_edu = st.selectbox(
    "Parental Education Level",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

math_score = st.number_input("Math Score", 0, 100, 70)
reading_score = st.number_input("Reading Score", 0, 100, 72)
writing_score = st.number_input("Writing Score", 0, 100, 74)

# Convert categorical inputs to numeric encoding (must match training preprocessing)
def preprocess_inputs():
    gender_map = {"male": 0, "female": 1}
    parent_map = {
        "some high school": 0, "high school": 1, "some college": 2,
        "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
    }
    lunch_map = {"free/reduced": 0, "standard": 1}
    test_map = {"none": 0, "completed": 1}

    return np.array([[
        gender_map[gender],
        parent_map[parent_edu],
        lunch_map[lunch],
        test_map[test_prep],
        math_score,
        reading_score,
        writing_score
    ]])

if st.button("🔮 Predict Performance"):
    features = preprocess_inputs()
    prediction = model.predict(features)

    st.success(f"✅ Predicted Final Performance Score: **{prediction[0]:.2f}** 🎯")

st.markdown("---")
st.caption("🚀 Built with Streamlit | Machine Learning for Student Success")
