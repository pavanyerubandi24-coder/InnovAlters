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
st.set_page_config(
    page_title="PulseGuard AI",
    page_icon="🩺",
    layout="wide"
)

DATA_PATH = "health_data.csv"  # Keep CSV in same folder

# =========================================================
# PREMIUM STYLISH MEDICAL UI
# =========================================================
st.markdown("""
<style>

/* GLOBAL */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding: 2rem 4rem;
}

/* HEADER */
.app-header {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg, #FFA500, #FF6A00);
    -webkit-background-clip: text;
    -webkit-text-fill-color:blue;
}

/* GLASS CARD */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
}

/* ================= ORANGE INPUT FIELDS ================= */

/* Number input + Selectbox container */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: white !important;
    color:black !important;
    border-radius: 10px !important;
    border: 2px solid #FFA500 !important;
}

/* Dropdown menu background */
div[role="listbox"] {
    background-color: #FF8C00 !important;
    color:white !important;
}

/* Input labels */
label {
    color: #FFA500 !important;
    font-weight: 600;
}

/* Focus effect */
input:focus {
    border: 2px solid #FFD580 !important;
    box-shadow: 0 0 10px #FFA500 !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #FFA500, #FF6A00);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    height: 3.2em;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #FFA500;
}
            h1
            {
            color:blue;}

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
# MODEL TRAINING
# =========================================================
@st.cache_resource
def train_model():
    if not os.path.exists(DATA_PATH):
        st.error("❌ health_data.csv not found in project folder.")
        return None, None

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required_cols = [
        "age","gender","height","weight",
        "cholesterol","gluc","smoke","alco","active",
        "ap_hi","ap_lo"
    ]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing column: {col}")
            return None, None

    if df["age"].max() > 200:
        df["age"] = df["age"] / 365

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
# HYBRID MEDICAL LOGIC
# =========================================================
def hybrid_adjustment(age, bmi, smoke, chol, gluc, alco, active):

    systolic = 100 + (age * 0.5) + (bmi * 0.8)
    diastolic = 60 + (age * 0.3) + (bmi * 0.5)

    if smoke:
        systolic += 8; diastolic += 5
    if chol:
        systolic += 5
    if gluc:
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
    st.markdown("<div class='app-header'>🩺 PulseGuard AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>AI Clinical Blood Pressure Intelligence</div>", unsafe_allow_html=True)

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

    st.markdown("<div class='app-header'>🩺 PulseGuard AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>AI-Powered Clinical Blood Pressure Estimation</div>", unsafe_allow_html=True)

    # Dashboard Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("System Status", "Active 🟢")
    m2.metric("Model Type", "Random Forest")
    m3.metric("Prediction Engine", "Hybrid AI")

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
            chol!="Normal",
            gluc!="Normal",
            alco=="Yes",
            active=="Yes"
        )

        systolic = round((ml_pred[0][0] + adj_sys)/2)
        diastolic = round((ml_pred[0][1] + adj_dia)/2)

        if systolic >= 140 or diastolic >= 90:
            color = "#ff4b5c"
            status = "HIGH BLOOD PRESSURE"
        elif systolic < 90 or diastolic < 60:
            color = "#4dabf7"
            status = "LOW BLOOD PRESSURE"
        else:
            color = "#51cf66"
            status = "NORMAL BLOOD PRESSURE"

        st.markdown(f"""
        <div class="result-box" style="border-top:8px solid {color};">
            <h2>Predicted Blood Pressure</h2>
            <h1 style="font-size:60px; color:{color};">
                {systolic} / {diastolic} mmHg
            </h1>
            <h3 style="color:{color};">{status}</h3>
            <p>BMI: {round(bmi,1)}</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption("⚠ This AI system provides predictive estimates and does not replace professional medical diagnosis.")

# =========================================================
# MAIN FLOW
# =========================================================
if not st.session_state.authenticated:
    login()
else:
    predict()
