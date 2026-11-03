
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

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section",
    ["Dashboard", "Dataset", "Sample Risk Prediction"]
)

if page == "Dashboard":
    st.header("Healthcare Data Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Age", f"{df['Age'].mean():.1f}")
    col3.metric("Average Glucose", f"{df['Glucose'].mean():.1f}")

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=7, edgecolor="black")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Records")
    ax.set_title("Distribution of Ages")
    ax.set_ylim(bottom=0)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Blood Pressure vs Glucose")
    fig, ax = plt.subplots()
    ax.scatter(
        df["Systolic_BP"],
        df["Glucose"],
        s=70,
        alpha=0.8
    )
    ax.set_xlabel("Systolic Blood Pressure")
    ax.set_ylabel("Glucose")
    ax.set_title("Blood Pressure vs Glucose")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

elif page == "Dataset":
    st.header("Sample Healthcare Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Records by Sample Label")
    st.bar_chart(df["Risk_Label"].value_counts().sort_index())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Sample Dataset",
        data=csv,
        file_name="healthcare_sample.csv",
        mime="text/csv"
    )

else:
    st.header("Sample Machine Learning Demonstration")
    st.write(
        "A Random Forest model trained on a small synthetic dataset."
    )

    X = df[["Age", "Systolic_BP", "Glucose"]]
    y = df["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.metric("Demo Model Accuracy", f"{accuracy:.0%}")

    st.warning(
        "This score is based on a tiny synthetic dataset. "
        "It does not measure real-world medical accuracy."
    )

    st.subheader("Try Sample Input Values")

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
            st.write("Sample model output: Label 1")
        else:
            st.write("Sample model output: Label 0")

        st.caption(
            "This is an illustrative model label, not a medical "
            "risk assessment or diagnosis."
        )

st.divider()
st.caption("Smart Healthcare Analytics | Student Project")
