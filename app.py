import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("student_performance_model.pkl")

# Streamlit page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🎓 Student Performance Prediction App")
st.markdown("This app predicts a student’s **Math Score** based on demographics, parental background, and test preparation.")

# Sidebar - User Input
st.sidebar.header("📝 Input Student Details")

gender = st.sidebar.selectbox("Gender", ["female", "male"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A","group B","group C","group D","group E"])
parent_edu = st.sidebar.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading = st.sidebar.slider("Reading Score", 0, 100, 70)
writing = st.sidebar.slider("Writing Score", 0, 100, 70)

# Collect data
input_data = pd.DataFrame({
    "gender": [gender],
    "race/ethnicity": [race],
    "parental level of education": [parent_edu],
    "lunch": [lunch],
    "test preparation course": [prep],
    "reading score": [reading],
    "writing score": [writing]
})

st.write("### 📋 Student Data Preview")
st.dataframe(input_data)

# Predict
if st.button("🔮 Predict Math Score"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Math Score: **{prediction:.2f}** / 100")

# Footer
st.markdown("---")
st.markdown("✨ Built with ❤️ using **Streamlit & Scikit-learn**")
