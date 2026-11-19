from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_approval_model.pkl"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to get a machine-learning prediction.")
st.warning("Educational demo only. This model uses a synthetic sample dataset and must not be used for real lending decisions.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("loan_form"):
    st.subheader("Applicant information")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c2:
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant monthly income", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant monthly income", min_value=0, value=0, step=500)
        credit_history = st.selectbox("Credit history", [1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)")
    with c3:
        loan_amount = st.number_input("Loan amount (in thousands)", min_value=1.0, value=120.0, step=10.0)
        loan_term = st.selectbox("Loan term (months)", [120, 180, 240, 300, 360])
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    submitted = st.form_submit_button("Predict loan status", type="primary")

if submitted:
    applicant = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area
    }])
    prediction = model.predict(applicant)[0]
    probabilities = model.predict_proba(applicant)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index] * 100

    if prediction == "Y":
        st.success(f"Demo prediction: likely approved ({confidence:.1f}% model confidence)")
    else:
        st.error(f"Demo prediction: likely rejected ({confidence:.1f}% model confidence)")
    st.caption("This is a model estimate, not a real bank decision or a guarantee of approval.")
