import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from catboost import Pool
import joblib
import warnings

# We need to add the root to the path for Streamlit to find 'src'
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline.predict_pipeline import PredictPipeline

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="bml1 | Strategic Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Strategic Customer Segmentation")
st.markdown("This page segments your *entire* customer base into four strategic groups. This shows you who to call, who to nurture, and who to ignore.")

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
def get_segmentation_data(_pipeline: PredictPipeline, raw_test_df: pd.DataFrame):
    # feature engineering from predict_pipeline
    df_engineered = _pipeline._feature_engineer(raw_test_df.drop('y', axis=1))

    num_cols = _pipeline.ALL_NUMERIC_COLS
    df_engineered[num_cols] = _pipeline.preprocessor.transform(df_engineered[num_cols])
    cat_cols = _pipeline.ALL_CATEGORICAL_COLS
    all_cols = num_cols + cat_cols
    df_processed = df_engineered[all_cols]
    
    predict_pool = Pool(data=df_processed, cat_features=cat_cols)
    
    probabilities = _pipeline.model.predict_proba(predict_pool)[:, 1]
    
    df_segments = pd.DataFrame({'probability': probabilities})
    
    # 7. Define the 4 buckets (as planned)
    # We use our 0.71 threshold for Platinum
    conditions = [
        (df_segments['probability'] >= 0.71),
        (df_segments['probability'] >= 0.50) & (df_segments['probability'] < 0.71),
        (df_segments['probability'] >= 0.20) & (df_segments['probability'] < 0.50),
        (df_segments['probability'] < 0.20)
    ]
    choices = ['Platinum', 'Gold', 'Silver', 'Exclusion']
    
    df_segments['Segment'] = np.select(conditions, choices, default='Other')
    
    return df_segments
# create segment
with st.spinner("Loading pipeline and segmenting customer base..."):
    pipeline = load_pipeline()
    raw_test_df = load_data()
    df_segments = get_segmentation_data(pipeline, raw_test_df)

segment_counts = df_segments['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']
# pie chart
st.header("Total Customer Base Segmentation")

fig = px.pie(
    segment_counts, 
    names='Segment', 
    values='Count',
    title='Distribution of Customer Segments',
    color='Segment',
    color_discrete_map={
        'Platinum': '#00b0f0', # Bright Blue
        'Gold': '#ffd700',     # Gold
        'Silver': "#dcdbdb",   # Silver similar somewhat
        'Exclusion': '#ff4b4b' # Red
    }
)
fig.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("Recommended Business Actions")
st.markdown("Based on the model's confidence, here is our recommended strategy for each group.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("🥇 Platinum (> 71%)")
    st.markdown("**(Sure Bets)**")
    st.info("**Action:** Call these customers immediately. They are highly likely to say 'yes'. Use your best salespeople for this high-value list.")

with col2:
    st.subheader("🥈 Gold (50% - 71%)")
    st.markdown("**(On the Fence)**")
    st.warning("**Action:** Do not call yet. Warm them up first with an automated email campaign. Follow up with a call *after* they've engaged.")

with col3:
    st.subheader("🥉 Silver (20% - 50%)")
    st.markdown("**(Unlikely)**")
    st.success("**Action:** Do not call. Add them to a low-cost, long-term newsletter. Nurture them for a future campaign in 6-12 months.")

with col4:
    st.subheader("❌ Exclusion (< 20%)")
    st.markdown("**(Confident 'No')**")
    st.error("**Action:** Do not contact. We are 95% confident they will say 'no'. Save your budget and protect your brand from annoyance.")