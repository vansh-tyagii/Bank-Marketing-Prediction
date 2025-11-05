# 🏦 Bank Marketing Optimization Platform

A complete, end-to-end **MLOps platform** for predicting bank term deposit subscriptions.  
This project goes far beyond a single model — it delivers a **full business intelligence suite** including a modular ML pipeline, a real-time API, and a 4-page Streamlit dashboard for ROI analysis and explainable AI (XAI).

---

## 📈 The Business Case: Why 48.5% PR-AUC is a 4x Victory

After rigorous experimentation (XGBoost, ANN, and CatBoost with advanced feature engineering), this project discovered a firm signal ceiling of **~48.5% PR-AUC** in the dataset.

This is not a “poor” score — it’s a **profitable business solution** when used strategically.

This platform transforms that insight into real business impact through two key mechanisms:

### 🚀 The Profit Engine (4x Lift)
The model is **4× more efficient than random calling**.  
By targeting only the top 10% of leads ranked by the model:
- The bank captures ~40% of all potential “yes” customers  
- While saving **90% of its call center budget**

### 💸 The Savings Engine (95% Precision)
The model identifies **“Confident No”** customers with **95% precision**.  
This enables the business to **save ~80% of campaign costs** by not calling low-value prospects.

> 🔹 This project demonstrates how to turn a “low academic score” into a **high-ROI business strategy**.

---

## ✨ The “Marketing OS” Dashboard

The heart of this platform is a **4-page Streamlit dashboard** that transforms ML outputs into business strategy.

### 1. 🚀 Campaign HQ (ROI & A/B Simulator)
**Purpose:** The main management dashboard.  
Features:
- **Before vs After Visuals:** Compare campaign cost and profit with vs without the model.  
- **A/B Test Simulator:** See performance of 1,000 random vs 1,000 model-ranked calls.  
- **Dual Pitch Engine:** Live-calculates both *Profit* and *Savings* engines based on user inputs.

---

### 2. 🔮 Customer Predictor (XAI)
**Purpose:** Real-time predictions with transparency.  
Features:
- **Instant Prediction:** Get a 0/1 subscription likelihood and probability.  
- **Explainability via SHAP:** Visualizes feature contributions with a waterfall plot (e.g., `poutcome='success'` +40%).  
- Builds **manager trust** through explainable AI.

---

### 3. 📊 Strategic Segmentation
**Purpose:** Turn predictions into actionable business strategy.  
Segments:
- 🥇 **Platinum (>71%)** — Call Now  
- 🥈 **Gold (50–71%)** — Nurture via Email  
- 🥉 **Silver (20–50%)** — Low-Cost Newsletter  
- ❌ **Exclusion (<20%)** — Do Not Call (Save Budget)

Gives marketers clear next actions per customer group.

---

### 4. 📈 MLOps Monitoring
**Purpose:** Prototype for model and data health tracking.  
Features:
- **Model Performance:** Tracks PR-AUC & ROC-AUC over time for concept drift.  
- **Data Drift Alerts:** Monitors economic indicators (e.g., `euribor3m`) to flag distribution shifts.

---

## 🛠 Tech Stack & Architecture

| Layer | Technology |
|:------|:------------|
| **Model** | CatBoost (Champion after XGBoost/ANN comparison) |
| **Explainability** | SHAP |
| **Backend / API** | FastAPI, Uvicorn, Pydantic (+ pydantic-extra-types) |
| **Dashboard / UI** | Streamlit, Plotly |
| **ML / Data** | Scikit-learn, Pandas, NumPy |
| **Orchestration** | Modular Python scripts per pipeline stage |

---


---

## 🚀 How to Run This Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vansh-tyagii/Bank-Marketing-Prediction.git
cd Bank-Marketing-Prediction

2️⃣ Set Up the Environment

(Recommended: use Conda or virtualenv)

# Create environment
conda create -n bml1 python=3.10 -y
conda activate bml1

# Install dependencies
pip install -r requirements.txt

3️⃣ Run the Full Training Pipeline

Recreates the champion CatBoost model and scaler inside artifacts/.

python -m src.pipeline.train_pipeline

4️⃣ Run the Backend (FastAPI)

Starts the REST API server for real-time predictions.

uvicorn app:app --Reload


🔹 Docs available at: http://127.0.0.1:8000/docs

5️⃣ Run the Frontend (Streamlit Dashboard)

Launch the interactive business dashboard.

streamlit run ui/dashboard.py

🔌 API Endpoint

POST /predict

Predicts subscription likelihood for a single customer.

Request Body (JSON) — matches BankCustomer schema
Response Example:

{
  "prediction": 1,
  "probability_of_yes": 0.8544
}


Threshold: 0.7111 → above this value = predicted “Yes”

🏁 Summary

✅ End-to-end MLOps system
✅ Business-driven ML with measurable ROI
✅ Explainable, monitorable, and production-ready

This project bridges the gap between academic model performance and real-world business value, proving that smart deployment and interpretation can turn even a modest-signal dataset into a profitable marketing intelligence engine.


---

Would you like me to make this version automatically include **GitHub-ready badges** (e.g., Python version, Streamlit, FastAPI, License, PR-AUC metric) at the top? It’ll make your repo’s header look more professional and “complete.”



