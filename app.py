import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction (Traditional ML) - Demo")

@st.cache_resource
def load_model():
    model = joblib.load("models/best_salary_model.joblib")
    return model

model = load_model()

st.sidebar.header("Input features")
experience = st.sidebar.slider("Years of Experience", min_value=0.0, max_value=40.0, value=3.0, step=0.1)
age = st.sidebar.number_input("Age", min_value=18, max_value=80, value=25)
education = st.sidebar.selectbox("Education", options=["High School","Bachelors","Masters","PhD"])
skill_level = st.sidebar.selectbox("Skill Level", options=["Beginner","Intermediate","Advanced","Expert"])

input_dict = {
    "Experience": [experience],
    "Age": [age],
    "Education": [education],
    "Skill_Level": [skill_level]
}
X_input = pd.DataFrame(input_dict)

st.subheader("Inputs")
st.table(X_input.T)

if st.button("Predict Salary"):
    try:
        pred = model.predict(X_input)[0]
        st.success(f"Predicted Salary: {pred:,.2f}")
    except Exception as e:
        st.error("Prediction failed: " + str(e))

    # small visualization
    xs = np.linspace(0, 40, 40)
    ys = [model.predict(pd.DataFrame({
        "Experience":[x],
        "Age":[age],
        "Education":[education],
        "Skill_Level":[skill_level]
    }))[0] for x in xs]
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel("Experience (years)")
    ax.set_ylabel("Predicted salary")
    st.pyplot(fig)
