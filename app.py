
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Smart Healthcare Analytics",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Smart Healthcare Data Analysis")
st.subheader("Data Visualization & Sample Risk Prediction")
st.info("Educational demonstration only. Not for medical diagnosis.")

@st.cache_data
def create_data():
    return pd.DataFrame({
        "Age": [25, 34, 45, 52, 61, 29, 48, 57, 39, 66,
                31, 43, 55, 70, 36, 50, 63, 27, 46, 59],
        "Systolic_BP": [115, 122, 135, 145, 155, 118, 138, 150,
                        128, 160, 120, 132, 148, 165, 125, 140,
                        158, 116, 136, 152],
        "Glucose": [85, 92, 110, 145, 165, 88, 120, 155, 105, 180,
                    90, 108, 150, 190, 98, 130, 170, 86, 115, 160],
        "Risk_Label": [0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                       0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    })

df = create_data()

st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choose a section",
    ["Dashboard", "Dataset", "Sample Risk Prediction"]
)

if page == "Dashboard":
    st.header("Healthcare Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Average Age", f"{df['Age'].mean():.1f}")
    c3.metric("Average Glucose", f"{df['Glucose'].mean():.1f}")

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=7)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Blood Pressure vs Glucose")
    st.scatter_chart(df, x="Systolic_BP", y="Glucose")

elif page == "Dataset":
    st.header("Sample Healthcare Dataset")
    st.dataframe(df, use_container_width=True)
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

else:
    st.header("Demonstration Model")

    X = df[["Age", "Systolic_BP", "Glucose"]]
    y = df["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.metric("Demo Model Accuracy", f"{accuracy:.0%}")

    st.warning(
        "This accuracy comes from a tiny synthetic dataset. "
        "It does not represent real-world medical performance."
    )

    st.subheader("Enter Sample Values")
    age = st.slider("Age", 18, 90, 40)
    bp = st.slider("Systolic Blood Pressure", 80, 200, 120)
    glucose = st.slider("Glucose", 60, 250, 100)

    if st.button("Run Demonstration"):
        sample = pd.DataFrame({
            "Age": [age],
            "Systolic_BP": [bp],
            "Glucose": [glucose]
        })
        result = model.predict(sample)[0]

        if result == 1:
            st.write("The sample model classified this record as label 1.")
        else:
            st.write("The sample model classified this record as label 0.")

        st.caption(
            "These labels are illustrative only and must not be "
            "used to assess an actual person's health."
        )

st.divider()
st.caption("Smart Healthcare Analytics | Student Project")
