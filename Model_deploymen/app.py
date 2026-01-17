import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("xgb_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("🏦 Loan Default Prediction App")
st.write("This app predicts whether a loan applicant is **Defaulter** or **Non-Defaulter**.")

# --- Mapping dictionaries for categorical features ---
gender_map = {"Male": 0, "Female": 1, "Joint": 2, "Not Available": 3}
loan_limit_map = {"Not Limited": 0, "Limited": 1}
approv_in_adv_map = {"No": 0, "Yes": 1}
loan_type_map = {"Type 1": 0, "Type 2": 1, "Type 3": 2}
loan_purpose_map = {"Home Purchase": 0, "Refinance": 1, "Improvement": 2, "Other": 3}
credit_worthiness_map = {"Good": 0, "Bad": 1}
open_credit_map = {"No": 0, "Yes": 1}
business_or_commercial_map = {"No": 0, "Yes": 1}
neg_amort_map = {"No": 0, "Yes": 1}
interest_only_map = {"No": 0, "Yes": 1}
lump_sum_map = {"No": 0, "Yes": 1}
construction_type_map = {"Type 1": 0, "Type 2": 1}
occupancy_type_map = {"Owner": 0, "Co-Owner": 1, "Tenant": 2}
secured_by_map = {"Home": 0, "Other": 1}
total_units_map = {"1 Unit": 0, "2 Units": 1, "3 Units": 2, "4 Units": 3}
credit_type_map = {"Conventional": 0, "FHA": 1, "VA": 2, "Other": 3}
co_applicant_credit_map = {"No": 0, "Yes": 1}
age_map = {"<25": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55-64": 4, "65-74": 5, "75+": 6}
submission_map = {"Not Submitted": 0, "Submitted": 1}
region_map = {"North": 0, "South": 1, "East": 2, "West": 3}
security_type_map = {"Type 1": 0, "Type 2": 1}

# --- User inputs ---
st.sidebar.header("Applicant Details")

loan_limit = loan_limit_map[st.sidebar.selectbox("Loan Limit", list(loan_limit_map.keys()))]
gender = gender_map[st.sidebar.selectbox("Gender", list(gender_map.keys()))]
approv_in_adv = approv_in_adv_map[st.sidebar.selectbox("Approval in Advance", list(approv_in_adv_map.keys()))]
loan_type = loan_type_map[st.sidebar.selectbox("Loan Type", list(loan_type_map.keys()))]
loan_purpose = loan_purpose_map[st.sidebar.selectbox("Loan Purpose", list(loan_purpose_map.keys()))]
credit_worthiness = credit_worthiness_map[st.sidebar.selectbox("Credit Worthiness", list(credit_worthiness_map.keys()))]
open_credit = open_credit_map[st.sidebar.selectbox("Open Credit", list(open_credit_map.keys()))]
business_or_commercial = business_or_commercial_map[st.sidebar.selectbox("Business/Commercial", list(business_or_commercial_map.keys()))]

loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=10000, max_value=5000000, step=5000)
rate_of_interest = st.sidebar.number_input("Rate of Interest (%)", min_value=0.0, max_value=20.0, step=0.1)
interest_rate_spread = st.sidebar.number_input("Interest Rate Spread", min_value=-5.0, max_value=10.0, step=0.1)
upfront_charges = st.sidebar.number_input("Upfront Charges", min_value=0.0, max_value=100000.0, step=100.0)
term = st.sidebar.number_input("Loan Term (months)", min_value=12, max_value=480, step=12)

neg_amort = neg_amort_map[st.sidebar.selectbox("Negative Amortization", list(neg_amort_map.keys()))]
interest_only = interest_only_map[st.sidebar.selectbox("Interest Only", list(interest_only_map.keys()))]
lump_sum_payment = lump_sum_map[st.sidebar.selectbox("Lump Sum Payment", list(lump_sum_map.keys()))]

property_value = st.sidebar.number_input("Property Value (₹)", min_value=10000, max_value=20000000, step=5000)
construction_type = construction_type_map[st.sidebar.selectbox("Construction Type", list(construction_type_map.keys()))]
occupancy_type = occupancy_type_map[st.sidebar.selectbox("Occupancy Type", list(occupancy_type_map.keys()))]
secured_by = secured_by_map[st.sidebar.selectbox("Secured By", list(secured_by_map.keys()))]
total_units = total_units_map[st.sidebar.selectbox("Total Units", list(total_units_map.keys()))]

income = st.sidebar.number_input("Applicant Income (₹)", min_value=0, max_value=200000, step=1000)
credit_type = credit_type_map[st.sidebar.selectbox("Credit Type", list(credit_type_map.keys()))]
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, step=1)
co_applicant_credit_type = co_applicant_credit_map[st.sidebar.selectbox("Co-Applicant Credit Type", list(co_applicant_credit_map.keys()))]
age = age_map[st.sidebar.selectbox("Age Group", list(age_map.keys()))]
submission_of_application = submission_map[st.sidebar.selectbox("Submission of Application", list(submission_map.keys()))]

ltv = st.sidebar.number_input("Loan-to-Value Ratio (LTV)", min_value=0.0, max_value=200.0, step=0.1)
region = region_map[st.sidebar.selectbox("Region", list(region_map.keys()))]
security_type = security_type_map[st.sidebar.selectbox("Security Type", list(security_type_map.keys()))]
dtir1 = st.sidebar.number_input("DTI Ratio (%)", min_value=0.0, max_value=100.0, step=1.0)

# --- Collect features into DataFrame ---
input_data = pd.DataFrame([[
    loan_limit, gender, approv_in_adv, loan_type, loan_purpose,
    credit_worthiness, open_credit, business_or_commercial,
    loan_amount, rate_of_interest, interest_rate_spread, upfront_charges, term,
    neg_amort, interest_only, lump_sum_payment,
    property_value, construction_type, occupancy_type, secured_by, total_units,
    income, credit_type, credit_score, co_applicant_credit_type, age, submission_of_application,
    ltv, region, security_type, dtir1
]], columns=[
    'loan_limit','Gender','approv_in_adv','loan_type','loan_purpose',
    'Credit_Worthiness','open_credit','business_or_commercial',
    'loan_amount','rate_of_interest','Interest_rate_spread','Upfront_charges','term',
    'Neg_ammortization','interest_only','lump_sum_payment',
    'property_value','construction_type','occupancy_type','Secured_by','total_units',
    'income','credit_type','Credit_Score','co-applicant_credit_type','age','submission_of_application',
    'LTV','Region','Security_Type','dtir1'
])

# --- Prediction ---
if st.button("🔮 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        st.success("✅ Prediction: Applicant is **Non-Defaulter**")
    else:
        st.error("❌ Prediction: Applicant is **Defaulter**")
