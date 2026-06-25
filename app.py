import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Custom CSS styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #d5f5e3;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .stApp {
        background-color: #FFDAB9;
    }
    .stApp * {
        color: black !important;
    }
     .stButton > button {
    color: white !important;
    background-color: #2d6a4f !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
}       
    </style>
""", unsafe_allow_html=True)

# Load and train model
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
Y = df['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)
predictions_test = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions_test)

# App Title
st.title("🏥 Diabetes Prediction System")
st.info(f"🎯 Model Accuracy: {round(accuracy * 100, 2)}%")
st.write("Enter patient details in the sidebar to predict diabetes!")

# Sidebar
st.sidebar.header("📋 Patient Details")
st.sidebar.write("---")

pregnancies = st.sidebar.slider("🤰 Pregnancies", 0, 17, 3)
st.sidebar.caption("Number of times pregnant")

glucose = st.sidebar.slider("🩸 Glucose Level", 1, 199, 120)
st.sidebar.caption("Normal: 70–140 mg/dl")

blood_pressure = st.sidebar.slider("💓 Blood Pressure", 1, 122, 72)
st.sidebar.caption("Normal: 60–80 mmHg")

skin_thickness = st.sidebar.slider("📏 Skin Thickness", 1, 99, 29)
st.sidebar.caption("Triceps skin fold thickness (mm)")

insulin = st.sidebar.slider("💉 Insulin Level", 1, 846, 100)
st.sidebar.caption("Normal: 16–166 mu U/ml")

bmi = st.sidebar.slider("⚖️ BMI", 1.0, 67.1, 32.4)
st.sidebar.caption("Normal BMI: 18.5–24.9")

dpf = st.sidebar.slider("🧬 Diabetes Pedigree Function", 0.08, 2.42, 0.47)
st.sidebar.caption("Genetic diabetes risk score")

age = st.sidebar.slider("🎂 Age", 1, 100, 25)
st.sidebar.caption("Patient's age in years")

st.sidebar.write("---")

# Predict Button
if st.sidebar.button("🔍 Predict Now"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi,
                            dpf, age]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ This patient is likely DIABETIC!")
        st.warning("💡 Health Tips:")
        st.write("🥗 Follow a low sugar diet")
        st.write("🏃 Exercise at least 30 minutes daily")
        st.write("💊 Consult a doctor immediately")
        st.write("💧 Drink plenty of water")
        st.write("⚖️ Maintain a healthy weight")
    else:
        st.success("✅ This patient is likely NOT DIABETIC!")
        st.info("💡 Stay Healthy Tips:")
        st.write("🥦 Eat balanced nutritious food")
        st.write("🏋️ Stay physically active")
        st.write("🩺 Get regular health checkups")
        st.write("😴 Sleep 7-8 hours daily")