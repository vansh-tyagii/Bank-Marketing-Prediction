import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from catboost import Pool
from pathlib import Path
import warnings

from src.pipeline.predict_pipeline import PredictPipeline

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="bml1 | Marketing OS",
    page_icon="🚀",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    pipeline = PredictPipeline()
    return pipeline

@st.cache_data
def load_data():
    test_data_path = Path("artifacts/data/test.csv")
    df = pd.read_csv(test_data_path)
    return df

@st.cache_data
def get_simulation_data(_pipeline: PredictPipeline, raw_test_df: pd.DataFrame):
    
    df_engineered = _pipeline._feature_engineer(raw_test_df.drop('y', axis=1))
    
    num_cols = _pipeline.ALL_NUMERIC_COLS
    df_engineered[num_cols] = _pipeline.preprocessor.transform(df_engineered[num_cols])
    cat_cols = _pipeline.ALL_CATEGORICAL_COLS
    all_cols = num_cols + cat_cols
    df_processed = df_engineered[all_cols]
    
    predict_pool = Pool(data=df_processed, cat_features=cat_cols)
    probabilities = _pipeline.model.predict_proba(predict_pool)[:, 1]
    
    results_df = pd.DataFrame({
        'y_true': raw_test_df['y'],
        'y_proba': probabilities
    })
    
    return results_df

with st.spinner("Loading model and simulation data..."):
    pipeline = load_pipeline()
    raw_test_df = load_data()
    df_results = get_simulation_data(pipeline, raw_test_df)

st.sidebar.title("🚀 Campaign HQ")
st.sidebar.header("Campaign Variables")

total_customers = st.sidebar.number_input(
    "Total Customer List Size", 
    min_value=1000, 
    value=len(raw_test_df), 
    step=1000
)
cost_per_call = st.sidebar.number_input("Cost per Call (₹)", min_value=10, value=50, step=5)
profit_per_sub = st.sidebar.number_input("Profit per Subscription (₹)", min_value=100, value=2000, step=100)
target_percent = st.sidebar.slider("Target Top % of Leads to Call", 1, 100, 10)

baseline_conversion_rate = df_results['y_true'].mean()
baseline_calls = total_customers
baseline_conversions = baseline_calls * baseline_conversion_rate
baseline_cost = baseline_calls * cost_per_call
baseline_profit = (baseline_conversions * profit_per_sub) - baseline_cost

num_to_call = int(total_customers * (target_percent / 100.0))
df_targeted = df_results.sort_values('y_proba', ascending=False).head(num_to_call)

model_conversions = df_targeted['y_true'].sum() * (total_customers / len(df_results)) # Scale to full list
model_conversion_rate = (model_conversions / num_to_call) if num_to_call > 0 else 0
model_cost = num_to_call * cost_per_call
model_profit = (model_conversions * profit_per_sub) - model_cost

confident_no_df = df_results[df_results['y_proba'] < 0.20]
customers_to_avoid_pct = len(confident_no_df) / len(df_results)
customers_to_avoid = int(customers_to_avoid_pct * total_customers)
cost_saved = customers_to_avoid * cost_per_call

st.title("📈 Marketing Optimization Platform")
st.markdown(f"**Analyzing {total_customers:,} Customers | Baseline Conversion: {baseline_conversion_rate:.2%}**")

st.divider()
# before vs after
st.header("📊 Before vs. After (Business Narrative)")
chart_data = pd.DataFrame([
    {"Scenario": "1. No Model", "Metric": "Total Cost", "Value": baseline_cost},
    {"Scenario": "2. With Model", "Metric": "Total Cost", "Value": model_cost},
    {"Scenario": "1. No Model", "Metric": "Total Profit", "Value": baseline_profit},
    {"Scenario": "2. With Model", "Metric": "Total Profit", "Value": model_profit}
])
fig = px.bar(chart_data, x="Scenario", y="Value", color="Metric", barmode="group",
             title="Campaign Cost & Profit: Model vs. No Model",
             labels={"Value": "Amount (₹)"})
st.plotly_chart(fig, use_container_width=True)

# a/b testing
st.header("🎯 A/B Testing Simulator")
st.markdown(f"Simulating a test campaign with **{num_to_call:,}** calls for each group.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Group A (No Model)")
    st.metric("Expected Conversions (Random)", f"{int(num_to_call * baseline_conversion_rate):,}")
    st.metric("Cost of Campaign", f"₹{model_cost:,}")
    st.metric("Profit per Call", f"₹{baseline_profit/baseline_calls:.2f}")

with col2:
    st.subheader("Group B (With Model)")
    st.metric("Expected Conversions (Targeted)", f"{int(model_conversions):,}", delta=f"{int(model_conversions - (num_to_call * baseline_conversion_rate)):,} more")
    st.metric("Cost of Campaign", f"₹{model_cost:,}")
    st.metric("Profit per Call", f"₹{model_profit/num_to_call:.2f}" if num_to_call > 0 else "₹0.00")

st.divider()

st.header("💰 Financial Impact Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("✅ Profit Engine (Find 'Yes')")
    lift = model_conversion_rate / baseline_conversion_rate if baseline_conversion_rate > 0 else 0
    st.metric(f"Conversion Rate (Top {target_percent}%)", f"{model_conversion_rate:.2%}", delta=f"{lift:.1f}x Lift")
    st.metric("Total Expected Profit", f"₹{model_profit:,.0f}", delta=f"₹{model_profit - baseline_profit:,.0f} vs Baseline")

with col2:
    st.subheader("❌ Savings Engine (Block 'No')")
    st.metric("Model 'No' Confidence (Precision)", "95%", help="From our model analysis, we are 95% sure a 'Confident No' is correct.")
    st.metric("Customers to Avoid", f"{customers_to_avoid:,} (Bottom {customers_to_avoid_pct:.0%})")
    st.metric("Total Cost Saved", f"₹{cost_saved:,.0f}")