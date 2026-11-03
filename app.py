
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= PAGE CONFIGURATION =================

st.set_page_config(
    page_title="Smart Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= COLOURFUL DESIGN =================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #10213a, #19152f);
    color: #f5f7ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151f3b, #10172b);
    border-right: 1px solid #29466d;
}
h1 {
    color: #42e8df !important;
    font-weight: 800 !important;
}
h2 {
    color: #80b9ff !important;
}
h3 {
    color: #d3a7ff !important;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #152d4c, #24244a);
    border: 1px solid #315b83;
    padding: 18px;
    border-radius: 15px;
}
[data-testid="stMetricLabel"] {
    color: #c4d5f2;
}
[data-testid="stMetricValue"] {
    color: #45e5dc;
}
.stButton > button {
    background: linear-gradient(90deg, #1769ff, #873bff);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 18px;
}
.stButton > button:hover {
    border: 1px solid #45e5dc;
    color: white;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #315b83;
    border-radius: 10px;
}
hr {
    border-color: #315b83;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================

st.title("🏥 Smart Healthcare Data Analysis")
st.subheader("📊 Data Visualization & Sample Risk Prediction")

st.markdown(
    "Explore sample healthcare data using Python, data visualization "
    "and machine learning."
)

st.info(
    "Educational demonstration only. This application is not "
    "intended for medical diagnosis or treatment."
)

# ================= SAMPLE DATASET =================

@st.cache_data
def create_dataset():
    return pd.DataFrame({
        "Age": [
            25, 34, 45, 52, 61, 29, 48, 57, 39, 66,
            31, 43, 55, 70, 36, 50, 63, 27, 46, 59
        ],
        "Systolic_BP": [
            115, 122, 135, 145, 155, 118, 138, 150, 128, 160,
            120, 132, 148, 165, 125, 140, 158, 116, 136, 152
        ],
        "Glucose": [
            85, 92, 110, 145, 165, 88, 120, 155, 105, 180,
            90, 108, 150, 190, 98, 130, 170, 86, 115, 160
        ],
        "Risk_Label": [
            0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
            0, 0, 1, 1, 0, 0, 1, 0, 0, 1
        ]
    })

df = create_dataset()

# ================= MODEL TRAINING =================

@st.cache_resource
def train_model():
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
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy

model, accuracy = train_model()

# ================= NAVIGATION =================

st.sidebar.title("💙 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "🏠 Dashboard",
        "📁 Dataset Explorer",
        "🤖 Sample Risk Prediction",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Smart Healthcare Analytics")
st.sidebar.caption("Student Machine Learning Project")

# ================= DASHBOARD =================

if page == "🏠 Dashboard":

    st.header("✨ Healthcare Data Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("📋 Total Records", len(df))
    c2.metric("👤 Average Age", f"{df['Age'].mean():.1f}")
    c3.metric("🧪 Average Glucose", f"{df['Glucose'].mean():.1f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📊 Age Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f4f7ff")

        ax.hist(
            df["Age"],
            bins=7,
            color="#29b6f6",
            edgecolor="#ffffff"
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Records")
        ax.set_title("Distribution of Ages")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("💗 Blood Pressure vs Glucose")

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f4f7ff")

        ax.scatter(
            df["Systolic_BP"],
            df["Glucose"],
            color="#e83e8c",
            edgecolor="#ffffff",
            s=65,
            alpha=0.85
        )

        ax.set_xlabel("Systolic Blood Pressure")
        ax.set_ylabel("Glucose")
        ax.set_title("Sample Measurement Comparison")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("📌 Dataset Highlights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Average Systolic Blood Pressure**")
        st.write(f"{df['Systolic_BP'].mean():.1f}")

    with col2:
        st.write("**Sample Records with Label 1**")
        st.write(int((df["Risk_Label"] == 1).sum()))

    st.caption(
        "All records are synthetic examples created for demonstration."
    )

# ================= DATASET EXPLORER =================

elif page == "📁 Dataset Explorer":

    st.header("📁 Sample Healthcare Dataset")

    st.write(
        "Explore the sample records used by this educational application."
    )

    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("🏷️ Records by Sample Label")

    label_counts = (
        df["Risk_Label"]
        .value_counts()
        .sort_index()
        .rename(index={0: "Label 0", 1: "Label 1"})
    )

    st.bar_chart(label_counts)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Sample Dataset",
        data=csv_data,
        file_name="healthcare_sample.csv",
        mime="text/csv"
    )

    st.warning(
        "Risk_Label is a demonstration label, not a clinically "
        "established diagnosis or health-risk category."
    )

# ================= SAMPLE PREDICTION =================

elif page == "🤖 Sample Risk Prediction":

    st.header("🤖 Machine Learning Demonstration")

    st.write(
        "A Random Forest classifier trained on a small synthetic dataset."
    )

    st.metric("Demo Model Accuracy", f"{accuracy:.0%}")

    st.warning(
        "The dataset contains only 20 synthetic records. The accuracy "
        "score is not evidence of real-world medical performance."
    )

    st.subheader("🎛️ Try Sample Input Values")

    age = st.slider("Age", 18, 90, 40)
    bp = st.slider("Systolic Blood Pressure", 80, 200, 120)
    glucose = st.slider("Glucose", 60, 250, 100)

    if st.button("🚀 Run Demonstration"):

        sample = pd.DataFrame({
            "Age": [age],
            "Systolic_BP": [bp],
            "Glucose": [glucose]
        })

        result = int(model.predict(sample)[0])

        st.subheader("Sample Model Output")

        if result == 1:
            st.success("The sample model returned Label 1.")
        else:
            st.success("The sample model returned Label 0.")

        st.caption(
            "This is an illustrative machine-learning label. It is not "
            "a medical risk assessment, diagnosis or treatment recommendation."
        )

        st.subheader("Model Feature Importance")

        importance = pd.Series(
            model.feature_importances_,
            index=["Age", "Systolic Blood Pressure", "Glucose"]
        ).sort_values()

        st.bar_chart(importance)

# ================= ABOUT PROJECT =================

else:

    st.header("ℹ️ About This Project")

    st.markdown("""
    ### 🎯 Project Objectives

    - Explore and summarize sample healthcare data.
    - Visualize data using charts and graphs.
    - Train a Random Forest classification model.
    - Evaluate the model on a held-out test sample.
    - Provide an interactive demonstration interface.

    ### 🛠️ Technologies Used

    - Python
    - Pandas
    - Matplotlib
    - Scikit-learn
    - Streamlit
    - GitHub

    ### 🔄 Project Workflow

    1. Prepare synthetic sample data.
    2. Explore the dataset.
    3. Create visualizations.
    4. Split data into training and test sets.
    5. Train and evaluate the model.
    6. Display a sample model output.

    ### ⚠️ Limitations

    This prototype uses only 20 synthetic records. Its labels and
    model accuracy are illustrative and have not been validated using
    real patient data. It must not be used for medical decisions.

    ### 👩‍💻 Author

    Linda Blessy J
    """)

# ================= FOOTER =================

st.divider()

st.markdown(
    "<div style='text-align:center; color:#8fa8cc;'>"
    "🏥 Smart Healthcare Analytics | Student Project"
    "</div>",
    unsafe_allow_html=True
)
