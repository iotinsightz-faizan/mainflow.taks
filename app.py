import streamlit as st
import pickle
import numpy as np

# Load model
with open("student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict math scores based on student details 🚀")

# Input fields
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.slider("Reading Score", 0, 100, 70)
writing_score = st.slider("Writing Score", 0, 100, 70)

# Encode categorical values manually (same as training encoding order)
gender_dict = {"male": 1, "female": 0}
race_dict = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
edu_dict = {
    "some high school": 0, "high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
}
lunch_dict = {"standard": 1, "free/reduced": 0}
prep_dict = {"none": 0, "completed": 1}

features = [
    gender_dict[gender],
    race_dict[race],
    edu_dict[parent_edu],
    lunch_dict[lunch],
    prep_dict[prep_course],
    reading_score,
    writing_score
]

# Prediction button
if st.button("🔮 Predict Math Score"):
    prediction = model.predict(np.array(features).reshape(1, -1))
    st.success(f"📊 Predicted Math Score: **{prediction[0]:.2f}**")
