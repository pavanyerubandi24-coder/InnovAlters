import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="PulseGuard AI", layout="wide")

DATA_PATH = "C:\pavan\health_data.csv"

# =========================================================
# PROFESSIONAL MEDICAL UI
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #F4F7FB; }
.block-container { padding: 2rem 3rem; }

.app-header {
    font-size: 36px;
    font-weight: 700;
    color: #0D47A1;
    margin-bottom: 5px;
}

.app-sub {
    font-size: 16px;
    color: #6B7280;
    margin-bottom: 30px;
}

.card {
    background: #FFFFFF;
    padding: 30px;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.result-box {
    background: #FFFFFF;
    padding: 45px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-top: 30px;
}

.stButton>button {
    background-color: #0D47A1;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    border: none;
}

.stButton>button:hover {
    background-color: #08306B;
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"] {
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "users" not in st.session_state:
    st.session_state.users = {"admin": "pulse123"}

# =========================================================
# TRAIN MODEL SAFELY
# =========================================================
@st.cache_resource
def train_model():

    if not os.path.exists(DATA_PATH):
        st.error("❌ cardio_train.csv not found in project folder.")
        return None, None

    # Load dataset safely
    df = pd.read_csv(DATA_PATH, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(DATA_PATH, sep=",")

    df.columns = df.columns.str.strip()

    required_cols = [
        "age","gender","height","weight",
        "cholesterol","gluc","smoke","alco","active",
        "ap_hi","ap_lo"
    ]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing column: {col}")
            st.write("Available columns:", df.columns.tolist())
            return None, None

    # Convert age from days to years
    if df["age"].max() > 200:
        df["age"] = df["age"] / 365

    # Clean unrealistic BP values
    df = df[(df["ap_hi"] > 70) & (df["ap_hi"] < 250)]
    df = df[(df["ap_lo"] > 40) & (df["ap_lo"] < 150)]

    features = [
        "age","gender","height","weight",
        "cholesterol","gluc","smoke","alco","active"
    ]

    X = df[features]
    y = df[["ap_hi","ap_lo"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler

model, scaler = train_model()

# =========================================================
# HYBRID REAL-WORLD LOGIC
# =========================================================
def hybrid_adjustment(age, bmi, smoke, chol, gluc, alco, active):

    systolic = 100 + (age * 0.5) + (bmi * 0.8)
    diastolic = 60 + (age * 0.3) + (bmi * 0.5)

    if smoke:
        systolic += 8; diastolic += 5
    if chol > 0:
        systolic += 5
    if gluc > 0:
        systolic += 5
    if alco:
        systolic += 5
    if active:
        systolic -= 5; diastolic -= 3

    return systolic, diastolic

# =========================================================
# LOGIN PAGE
# =========================================================
def login():

    st.markdown("<div class='app-header'>PulseGuard AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Clinical Blood Pressure Estimation System</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and \
           st.session_state.users[username] == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREDICTION PAGE
# =========================================================
def predict():

    if model is None:
        return

    st.markdown("<div class='app-header'>Blood Pressure Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Enter patient clinical parameters</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)",18,100)
        gender = st.selectbox("Gender",["Female","Male"])
        height = st.number_input("Height (cm)",120,220)
        weight = st.number_input("Weight (kg)",30,200)

    with col2:
        chol = st.selectbox("Cholesterol Level",["Normal","Above Normal","High"])
        gluc = st.selectbox("Glucose Level",["Normal","Above Normal","High"])
        smoke = st.selectbox("Smoker",["No","Yes"])
        alco = st.selectbox("Alcohol Consumption",["No","Yes"])
        active = st.selectbox("Physically Active",["No","Yes"])

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict Blood Pressure"):

        bmi = weight / ((height/100)**2)

        input_data = np.array([[age,
                                1 if gender=="Male" else 0,
                                height,
                                weight,
                                1 if chol=="Above Normal" else 2 if chol=="High" else 0,
                                1 if gluc=="Above Normal" else 2 if gluc=="High" else 0,
                                1 if smoke=="Yes" else 0,
                                1 if alco=="Yes" else 0,
                                1 if active=="Yes" else 0]])

        scaled = scaler.transform(input_data)
        ml_pred = model.predict(scaled)

        adj_sys, adj_dia = hybrid_adjustment(
            age, bmi,
            smoke=="Yes",
            1 if chol!="Normal" else 0,
            1 if gluc!="Normal" else 0,
            alco=="Yes",
            active=="Yes"
        )

        systolic = round((ml_pred[0][0] + adj_sys)/2)
        diastolic = round((ml_pred[0][1] + adj_dia)/2)

        # Category
        if systolic >= 140 or diastolic >= 90:
            color = "#C62828"
            status = "HIGH BLOOD PRESSURE"
        elif systolic < 90 or diastolic < 60:
            color = "#1565C0"
            status = "LOW BLOOD PRESSURE"
        else:
            color = "#2E7D32"
            status = "NORMAL BLOOD PRESSURE"

        st.markdown(f"""
        <div class="result-box" style="border-top:6px solid {color};">
            <h2>Predicted Blood Pressure</h2>
            <h1 style="color:{color}; font-size:48px;">{systolic} / {diastolic} mmHg</h1>
            <h3 style="color:{color};">{status}</h3>
            <p style="color:#6B7280;">BMI: {round(bmi,1)}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# MAIN FLOW
# =========================================================
if not st.session_state.authenticated:
    login()
else:
    predict()