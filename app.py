import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Smart Campus AI", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

data = {
    "cgpa": [9,8,7,6,5,9,4,7,8,6],
    "skills": [9,8,7,5,4,9,3,6,8,5],
    "internships": [2,1,1,0,0,3,0,1,2,0],
    "placed": [1,1,1,0,0,1,0,1,1,0]
}

df = pd.DataFrame(data)

X = df[["cgpa", "skills", "internships"]]
y = df["placed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

def predict_placement(cgpa, skills, internships):
    return model.predict([[cgpa, skills, internships]])[0]

menu = st.sidebar.selectbox(
    "Smart Campus AI",
    ["Home", "Placement Predictor", "Dashboard"]
)

st.sidebar.success(f"Model Accuracy: {acc*100:.2f}%")

if menu == "Home":
    st.title("Smart Campus AI System")
    st.write("AI-based Placement Prediction using Machine Learning")

if menu == "Placement Predictor":
    st.title("Placement Predictor")

    name = st.text_input("Enter Student Name")
    cgpa = st.number_input("CGPA")
    skills = st.number_input("Skills (0-10)")
    internships = st.number_input("Internships")

    if st.button("Predict"):
        result = predict_placement(cgpa, skills, internships)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "cgpa": cgpa,
            "result": result
        })

        st.write("### Student Profile")
        st.write("Name:", name)
        st.write("CGPA:", cgpa)
        st.write("Skills:", skills)
        st.write("Internships:", internships)

        if result == 1:
            st.success(f"{name} - HIGH CHANCE OF PLACEMENT")
            st.metric("Risk Level", "LOW")
        else:
            st.warning(f"{name} - LOW CHANCE OF PLACEMENT")
            st.metric("Risk Level", "HIGH")

            st.write("### Improvements")
            if cgpa < 6:
                st.write("Improve CGPA")
            if skills < 5:
                st.write("Improve Skills")
            if internships == 0:
                st.write("Do Internship")

if menu == "Dashboard":
    st.title("Live Prediction Dashboard")

    if len(st.session_state.history) == 0:
        st.warning("No predictions yet Go to Predictor page first")
    else:
        cgpa_data = [i["cgpa"] for i in st.session_state.history]
        result_data = [i["result"] for i in st.session_state.history]

        fig, ax = plt.subplots()
        ax.plot(cgpa_data, result_data, marker='o')

        ax.set_xlabel("CGPA")
        ax.set_ylabel("Placement Result (0/1)")
        ax.set_title("Real-Time Prediction History")

        st.pyplot(fig)