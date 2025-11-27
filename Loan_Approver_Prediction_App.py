import streamlit as st
import pickle
import pandas as pd

# --------------------------- #
# PAGE CONFIG
# --------------------------- #
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💸",
    layout="centered",
)

# --------------------------- #
# CUSTOM STYLING
# --------------------------- #
st.markdown("""
    <style>
        .main {
            background-color: #f1f5f9;
        }
        .title-box {
            background: linear-gradient(135deg, #3a5a40, #588157);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .section {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .predict-btn button {
            background-color: #3a5a40 !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 20px !important;
            padding: 10px 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------- #
# TITLE
# --------------------------- #
st.markdown("""
    <div class="title-box">
        <h1>💸 Loan Approval Prediction System</h1>
        <p>Predict loan approval manually or using bulk CSV upload</p>
    </div>
""", unsafe_allow_html=True)

# --------------------------- #
# LOAD MODEL & FEATURES
# --------------------------- #
model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("model_features.pkl", "rb"))

# --------------------------- #
# TABS FOR MANUAL / BULK
# --------------------------- #
tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Bulk File Upload"])

# =================================================================== #
# ==========================  MANUAL INPUT  ========================== #
# =================================================================== #
# =================================================================== #
# ===========================  MANUAL INPUT  ========================= #
# =================================================================== #

with tab1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📝 Manual Loan Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapp_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.selectbox("Loan Amount Term", [180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    if st.button("🔍 Predict Loan Approval"):
        input_dict = {
            "Gender": gender,
            "Married": married,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapp_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Dependents": dependents
        }

        df_manual = pd.DataFrame([input_dict])

        # One-hot encode
        df_manual_encoded = pd.get_dummies(df_manual)

        # Add missing columns
        for col in feature_names:
            if col not in df_manual_encoded.columns:
                df_manual_encoded[col] = 0

        df_manual_encoded = df_manual_encoded[feature_names]

        # Prediction & probability
        pred = model.predict(df_manual_encoded)[0]
        prob = model.predict_proba(df_manual_encoded)[0].max() * 100

        # Convert prediction text
        decision = "Approved" if pred == 1 else "Not Approved"

        # Explanation logic
        explanation = ""

        if credit_history == 1:
            explanation += "✔ Strong credit history. "
        else:
            explanation += "❌ Weak credit history. "

        total_income = applicant_income + coapp_income
        if total_income > 6000:
            explanation += "✔ High total income. "
        elif total_income > 3000:
            explanation += "✔ Moderate income. "
        else:
            explanation += "❌ Low income. "

        if loan_amount < 50000:
            explanation += "✔ Loan amount is affordable. "
        else:
            explanation += "❌ High loan amount compared to income. "

        # Badge UI
        badge_color = "green" if pred == 1 else "red"
        badge_html = f"""
        <div style='padding:10px;width:200px;text-align:center;
        background-color:{badge_color};color:white;border-radius:10px;
        font-size:20px; font-weight:bold;'>{decision}</div>
        """

        st.markdown("### 🟦 Prediction Result")
        st.markdown(badge_html, unsafe_allow_html=True)

        st.metric(label="Confidence Level", value=f"{prob:.2f}%")
        st.info(f"📝 **Reason:** {explanation}")

    st.markdown('</div>', unsafe_allow_html=True)

# =================================================================== #
# ==========================  BULK UPLOAD  =========================== #
# =================================================================== #
# =================================================================== #
# ==========================  BULK UPLOAD  =========================== #
# =================================================================== #
with tab2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📁 Upload CSV File")

    st.write("Upload a CSV file containing the same columns used during training.")

    file = st.file_uploader("Upload your CSV", type=["csv"])

    if file:
        df_bulk = pd.read_csv(file)
        st.info("📌 File Uploaded Successfully!")

        # One-hot encoding
        df_encoded = pd.get_dummies(df_bulk)

        # Add missing columns
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # Remove extra cols
        df_encoded = df_encoded[feature_names]

        # Predictions + probability
        preds = model.predict(df_encoded)
        probs = model.predict_proba(df_encoded)

        # Convert (0/1) to text
        decision_text = ["Approved" if p == 1 else "Not Approved" for p in preds]

        # Add to bulk df
        df_bulk["Loan_Status_Prediction"] = decision_text
        df_bulk["Confidence (%)"] = (probs.max(axis=1) * 100).round(2)

        # Explanations for each prediction
        explanations = []
        for i, row in df_bulk.iterrows():
            exp = ""

            if row["Credit_History"] == 1:
                exp += "✔ Good credit history. "
            else:
                exp += "❌ Poor credit history. "

            total_income = row["ApplicantIncome"] + row.get("CoapplicantIncome", 0)
            if total_income > 6000:
                exp += "✔ High combined income. "
            elif total_income > 3000:
                exp += "✔ Moderate income. "
            else:
                exp += "❌ Low income. "

            if row["LoanAmount"] < 50000:
                exp += "✔ Low loan amount. "
            else:
                exp += "❌ Higher loan amount. "

            explanations.append(exp)

        df_bulk["Reason"] = explanations

        # Green/red badge
        def color_badge(val):
            if val == "Approved":
                return '<span style="color: white; background-color: green; padding: 4px 12px; border-radius: 8px;">Approved</span>'
            else:
                return '<span style="color: white; background-color: red; padding: 4px 12px; border-radius: 8px;">Not Approved</span>'

        df_bulk["Prediction_Badge"] = df_bulk["Loan_Status_Prediction"].apply(color_badge)

        # Display UI table
        st.write("### ✅ Prediction Results")
        st.write(
            df_bulk.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

        # Download button
        csv_output = df_bulk.drop(columns=["Prediction_Badge"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv_output,
            file_name="Loan_Prediction_Results.csv",
            mime="text/csv"
        )

    st.markdown('</div>', unsafe_allow_html=True)

