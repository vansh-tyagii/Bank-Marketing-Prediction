import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os ,sys
from src.pipeline.predict_pipeline import PredictPipeline
# connect pages dont know what this line is but it connect pages folder 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


st.set_page_config(
    page_title="bml1 | XAI Predictor",
    page_icon="🔮",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    pipeline = PredictPipeline()
    return pipeline

with st.spinner("Loading AI model and explainers..."):
    pipeline = load_pipeline()

st.title("🔮 Customer Predictor & Explainability (XAI)")
st.markdown("Enter a customer's details to get a prediction and see *why* the model made its choice.")

with st.expander("Enter Customer Details Here"):
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Personal Info")
            age = st.number_input("Age", 18, 100, 45)
            job = st.selectbox("Job", ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'self-employed', 'entrepreneur', 'housemaid', 'unemployed', 'unknown'])
            marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
            education = st.selectbox("Education", ['basic.9y', 'university.degree', 'high.school', 'professional.course', 'basic.4y', 'basic.6y', 'illiterate', 'unknown'])

        with col2:
            st.subheader("Financial Info")
            default = st.selectbox("Credit in Default?", ['no', 'yes', 'unknown'])
            housing = st.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])
            loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
            
            st.markdown("---")
            st.markdown("*Economic context (uses defaults)*")
            emp_var_rate = st.number_input("Emp. Variation Rate", value=1.1, format="%.1f")
            cons_price_idx = st.number_input("Consumer Price Index", value=93.994, format="%.3f")
            cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, format="%.1f")
            euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, format="%.3f")
            nr_employed = st.number_input("Number of Employees", value=5191.0, format="%.1f")

        with col3:
            st.subheader("Campaign Info")
            contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
            month = st.selectbox("Last Contact Month", ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'])
            day_of_week = st.selectbox("Last Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
            campaign = st.number_input("Contacts This Campaign", 1, 50, 2)
            pdays = st.number_input("Days Since Last Contact (999=never)", 0, 999, 999)
            previous = st.number_input("Contacts Before This Campaign", 0, 10, 0)
            poutcome = st.selectbox("Previous Campaign Outcome", ['nonexistent', 'failure', 'success'])
        
        st.markdown("---")
        submit_button = st.form_submit_button("Analyze Customer")

if submit_button:
    raw_data = {
        'age': age, 'job': job.lower(), 'marital': marital.lower(), 'education': education.lower(),
        'default': default.lower(), 'housing': housing.lower(), 'loan': loan.lower(), 
        'contact': contact.lower(), 'month': month.lower(), 'day_of_week': day_of_week.lower(), 
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome.lower(),
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    
    with st.spinner("Analyzing customer..."):
        result = pipeline.predict(raw_data)
        
        shap_fig = pipeline.explain(raw_data)

        prediction_label = "YES" if result['prediction'] == 1 else "NO"
        prediction_icon = "✅" if prediction_label == "YES" else "❌"

        st.header(f"Model Prediction: {prediction_label} {prediction_icon}")
        
        col1, col2 = st.columns(2)
        col1.metric("Probability of 'Yes'", f"{result['probability_of_yes']:.2%}")
        col2.metric("Recommendation", "Add to Platinum List" if prediction_label == "YES" else "Add to Exclusion List")
        
        st.header("Why did the model make this decision?")
        st.pyplot(shap_fig, bbox_inches='tight')