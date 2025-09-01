# app_stylish.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Student Performance Predictor", layout="wide", initial_sidebar_state="expanded")

# ---- Sidebar ----
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Choose model", ["Random Forest", "Decision Tree", "Logistic Regression"])
show_eda = st.sidebar.checkbox("Show advanced EDA", True)
show_shap = st.sidebar.checkbox("Show SHAP explanations (optional)", False)
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ — Update UI as needed")

# ---- Load models ----
MODEL_MAP = {
    "Random Forest": "models/pipe_rf.pkl",
    "Decision Tree": "models/pipe_dt.pkl",
    "Logistic Regression": "models/pipe_lr.pkl"
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_MAP[model_choice])

# ---- Header ----
st.title("🎓 Student Performance Predictor — Stylish")
st.markdown("Predict whether a student will **PASS** (average ≥ 50) using demographics and scores. Use the sidebar to toggle EDA / SHAP.")

# ---- Input columns ----
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Student inputs")
    gender = st.selectbox("Gender", ["female","male"])
    race = st.selectbox("Race/Ethnicity", ["group A","group B","group C","group D","group E"])
    parent_edu = st.selectbox("Parental Level of Education", [
        "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
    ])
    lunch = st.selectbox("Lunch", ["standard","free/reduced"])
    prep = st.selectbox("Test Preparation Course", ["none","completed"])

    st.markdown("**Scores (0–100)**")
    math = st.slider("Math score", 0, 100, 60)
    reading = st.slider("Reading score", 0, 100, 60)
    writing = st.slider("Writing score", 0, 100, 60)

with col2:
    st.subheader("Quick overview")
    avg = round((math + reading + writing) / 3, 2)
    st.metric("Estimated Average", f"{avg}")
    st.info("Tip: Toggle advanced EDA in the sidebar to view plots and feature importances.")

# ---- Predict ----
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch,
        "test preparation course": prep,
        "math score": math,
        "reading score": reading,
        "writing score": writing
    }])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.success("✅ Prediction: PASS")
    else:
        st.error("❌ Prediction: FAIL")

    if proba is not None:
        st.write("Class probabilities:")
        st.write(pd.DataFrame([proba], columns=["Fail","Pass"]).T)

# ---- Advanced EDA ----
if show_eda:
    st.markdown("---")
    st.subheader("Advanced EDA & Model Insights")

    # load dataset (local)
    try:
        df = pd.read_csv("StudentsPerformance[1].csv")
    except Exception:
        st.warning("Dataset CSV not found in app folder. Upload 'StudentsPerformance[1].csv' or place it here.")
        df = None

    if df is not None:
        df['average_score'] = df[['math score','reading score','writing score']].mean(axis=1)
        # 1) correlation heatmap
        corr = df[['math score','reading score','writing score','average_score']].corr()
        fig, ax = plt.subplots(figsize=(4,3))
        cax = ax.matshow(corr, cmap='viridis')
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

        # 2) Average score by parental education (bar)
        pass_by_parent = df.groupby('parental level of education')['average_score'].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.bar(pass_by_parent.index, pass_by_parent.values)
        ax2.set_xticklabels(pass_by_parent.index, rotation=45, ha='right')
        ax2.set_ylabel("Average score")
        ax2.set_title("Average score by parental level of education")
        st.pyplot(fig2)

        # 3) Pass rate table
        pass_rates = df.groupby(['test preparation course'])['passed'].mean().sort_values(ascending=False)
        st.write("Pass rate by Test Preparation Course:")
        st.table(pass_rates.reset_index().rename(columns={'passed':'pass_rate'}))

        # 4) Model comparison quick metrics (on-the-fly using available models)
        st.markdown("#### Quick Model Comparison (on local dataset)")
        X_full = df[['gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score','writing score']]
        y_full = (df[['math score','reading score','writing score']].mean(axis=1) >= 50).astype(int)

        metrics_df = []
        for name, path in MODEL_MAP.items():
            try:
                m = joblib.load(path)
                y_pred = m.predict(X_full)
                acc = accuracy_score(y_full, y_pred)
                cm = confusion_matrix(y_full, y_pred)
                metrics_df.append({'model': name, 'accuracy': round(acc,4), 'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1])})
            except Exception as e:
                metrics_df.append({'model': name, 'accuracy': None, 'tn':None,'fp':None,'fn':None,'tp':None})
        st.table(pd.DataFrame(metrics_df).set_index('model'))

# ---- SHAP explanations (optional) ----
if show_shap:
    st.markdown("---")
    st.subheader("SHAP explanations (may be slow to compute)")
    try:
        import shap
        # use a small sample for speed
        df0 = pd.read_csv("StudentsPerformance[1].csv").sample(min(300, len(pd.read_csv("StudentsPerformance[1].csv"))), random_state=42)
        X0 = df0[['gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score','writing score']]
        # shap explainer expects model with predict_proba or have raw_tree support; we wrap using pipeline
        explainer = shap.Explainer(model.predict_proba, X0, feature_names=X0.columns)
        shap_values = explainer(X0)
        st.write("SHAP summary (beeswarm):")
        shap.plots.beeswarm(shap_values[:,: ,1], max_display=12)  # might render in notebook; Streamlit may require shap.plots._beeswarm
        st.write("Note: If SHAP plotting doesn't render, run offline or remove shap.")
    except Exception as e:
        st.write("SHAP not available or failed:", e)

st.markdown("---")
st.caption("Tip: Modify the app layout & texts in app_stylish.py as needed.")
