import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.utils.logger import logger
from catboost import Pool
import shap
import matplotlib.pyplot as plt

class PredictPipeline:

    def __init__(self):
        self.optimal_threshold = 0.71
        self.model_path = Path("artifacts/model.pkl")
        self.scaler_path = Path("artifacts/preprocessor.pkl")
        
        logger.info("Loading model and preprocessor objects...")
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.scaler_path)
        logger.info("Model and preprocessor loaded successfully.")

        self.explainer = shap.TreeExplainer(self.model)

        self.ALL_NUMERIC_COLS = [
            'age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
            'was_contacted_prev', 'poutcome_success', 'age_sq',
            'aggression_ratio', 'economic_mood', 'young_student', 'senior_retired',
            'success_rate_prev', 'month_num', 'day_num'
        ]

        self.ALL_CATEGORICAL_COLS = [
            'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome'
        ]

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Applies the same feature engineering as used in training.
            """
            df_new = df.copy()
            
            # feature engineering 
            df_new['was_contacted_prev'] = (df_new['pdays'] != 999).astype(int)
            top_jobs = ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'self-employed']
            df_new['job'] = df_new['job'].apply(lambda x: x if x in top_jobs else 'other')
            df_new['poutcome_success'] = (df_new['poutcome'] == 'success').astype(int)
            df_new['age_sq'] = df_new['age']**2
            df_new['aggression_ratio'] = df_new['campaign'] / (df_new['previous'] + 1)
            df_new['economic_mood'] = df_new['cons.price.idx'] / (df_new['cons.conf.idx'] + 1e-6)
            df_new['young_student'] = ((df_new['age'] < 25) & (df_new['job'] == 'student')).astype(int)
            df_new['senior_retired'] = ((df_new['age'] > 60) & (df_new['job'] == 'retired')).astype(int)
            df_new['success_rate_prev'] = df_new['poutcome_success'] / (df_new['previous'] + 1)
            month_order = {'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            df_new['month_num'] = df_new['month'].map(month_order).fillna(0)
            day_map = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5}
            df_new['day_num'] = df_new['day_of_week'].map(day_map).fillna(0)
            
            # fill missing
            for col in self.ALL_NUMERIC_COLS:
                df_new[col] = df_new[col].fillna("Missing")

            return df_new
    
    def _preprocess_data(self, data):
        
            if not isinstance (data, pd.DataFrame):
                data = pd.DataFrame([data])
            data_engineered = self._feature_engineer(data)
            data_engineered[self.ALL_NUMERIC_COLS] = self.preprocessor.transform(data_engineered[self.ALL_NUMERIC_COLS])
            
            all_cols = self.ALL_NUMERIC_COLS + self.ALL_CATEGORICAL_COLS
            data_processed = data_engineered[all_cols]

            prediction_pool = Pool(
                data=data_processed,
                cat_features=self.ALL_CATEGORICAL_COLS
            )
            return prediction_pool, data_processed

    def predict(self,data) -> dict:
            prediction_pool, _ = self._preprocess_data(data) 
            pred_probas = self.model.predict_proba(prediction_pool)
            
            if pred_probas.ndim == 2:
                prob_of_yes = pred_probas[0, 1]
            else:
                prob_of_yes = pred_probas[1]
            
            prediction_label = (prob_of_yes >= self.optimal_threshold).astype(int)
        # used in api
            result = {
                "prediction": int(prediction_label),
                "probability_of_yes": float(prob_of_yes)
            }

            logger.info(f"Prediction probability: {prob_of_yes}, Label: {prediction_label}")
            return result
         
         
    def explain(self, data):
        """
        Takes raw data and returns a SHAP waterfall plot.
        """
        try:
            logger.info("Starting explanation...")
            prediction_pool, data_processed = self._preprocess_data(data)
            
            shap_values = self.explainer.shap_values(prediction_pool)
            
            shap_explanation = shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=data_processed.iloc[0],
                feature_names=data_processed.columns.tolist()
            )
            
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_explanation, show=False)
            
            logger.info("SHAP plot created successfully.")
            return fig
            
        except Exception as e:
            logger.error(f"Error during explanation: {e}")
            raise e