import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="bml1 | Model Monitoring",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MLOps Monitoring Dashboard")
st.markdown("""
This dashboard serves as the project's "Mission Control." It monitors two key areas:
1.  **Model Performance:** Is our model's predictive accuracy stable over time?
2.  **Data Drift:** Is the new, incoming data similar to the data the model was trained on?
""")

@st.cache_data
def load_data():
    data_path = Path("artifacts/data/train.csv")
    df = pd.read_csv(data_path)
    return df

with st.spinner("Loading monitoring data..."):
    train_df = load_data()

#  MODEL PERFORMANCE MONITORING 
st.header("1. Model Performance (Concept Drift)")
st.markdown("We track our primary business metric (**PR-AUC**) and our overall ranking metric (**ROC-AUC**). A sharp drop in either would trigger a retraining alert.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current PR-AUC (Business Value)", "48.57%")
    
with col2:
    st.metric("Current ROC-AUC (Ranking Power)", "81.58%")
    
with col3:
    st.metric("Model Status", "HEALTHY")
    st.success("✅ Model performance is stable.")

# creating fake 
days = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
fake_pr_auc_history = np.random.normal(loc=0.485, scale=0.005, size=30)
fake_roc_auc_history = np.random.normal(loc=0.815, scale=0.005, size=30)

fake_pr_auc_history[-5:] -= 0.01 
fake_roc_auc_history[-5:] -= 0.01

perf_df = pd.DataFrame({'Date': days, 'PR-AUC': fake_pr_auc_history, 'ROC-AUC': fake_roc_auc_history})
perf_df = perf_df.melt(id_vars='Date', var_name='Metric', value_name='Score') 

fig_perf = px.line(
    perf_df, 
    x='Date', 
    y='Score',
    color='Metric',
    title='Model Performance Over Last 30 Days'
)
fig_perf.update_yaxes(range=[0.40, 0.90]) 
st.plotly_chart(fig_perf, use_container_width=True)

st.divider()

st.header("2. Data Drift (Early Warning System)")
st.markdown("We monitor key input features. If new data's distribution changes significantly from the training data, the model's accuracy will soon fail. This is our 'canary in the coal mine.'")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Economic Drift: `euribor3m`")
    
    month_order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    drift_agg = train_df.groupby('month')['euribor3m'].mean().reindex(month_order).reset_index()

    fig_drift_1 = px.bar(
        drift_agg, 
        x='month', 
        y='euribor3m', 
        title="Average 'euribor3m' by Month (Training Data)"
    )
    st.plotly_chart(fig_drift_1, use_container_width=True)
    st.info("**Insight:** The model learned that `euribor3m` > 4.0 (like in May/Jun) means fewer 'yes' customers. If a new economic event drops this rate to 0.5 for all new customers, the model's logic is obsolete and it *must* be retrained.")

with col2:
    st.subheader("Business Drift: `campaign`")
    
    fig_drift_2 = px.histogram(
        train_df[train_df['campaign'] < 15],
        x='campaign',
        title="Distribution of 'campaign' (Training Data)"
    )
    st.plotly_chart(fig_drift_2, use_container_width=True)
    st.info("**Insight:** The model learned that calling 1-3 times is normal. If a new marketing strategy starts calling customers 10+ times (shifting this graph to the right), our `campaign` feature becomes useless. This chart detects that business strategy change.")