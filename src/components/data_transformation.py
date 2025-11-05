import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.entity.config_entity import DataTransformationConfig
from src.config.configuration import ConfigurationManager
from src.utils.common import save_object
from src.utils.logger import logger
from pathlib import Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, data_ingestion_config):
        self.config = config
        self.data_ingestion_config = data_ingestion_config
        self.scalar_path = self.config.preprocessor_obj_file

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies our advanced feature engineering from our EDA and experiments.
        """
        logger.info("Applying ADVANCED feature engineering...")
        df_new = df.copy()

        df_new["was_contacted_prev"] = (df_new["pdays"] != 999).astype(int)
        top_jobs = [
            'admin.', 'blue-collar', 'technician', 'services','management',
            'retired', 'student', 'self-employed']
        df_new["job"] = df_new["job"].apply(lambda x: x if x in top_jobs else 'other')
        df_new["poutcome_success"] = (df_new["poutcome"] == "success").astype(int)
        df_new["age_sq"] = df_new["age"] ** 2
        df_new["aggression_ratio"] = df_new["campaign"] / (df_new["previous"] + 1)
        df_new["economic_mood"] = df_new["cons.price.idx"] / (df_new["cons.conf.idx"] + 1e-6)

        df_new["young_student"] = ((df_new["age"] < 25) & (df_new["job"] == "student")).astype(int)
        df_new["senior_retired"] = ((df_new["age"] > 60) & (df_new["job"] == "retired")).astype(int)
        df_new["success_rate_prev"] = df_new["poutcome_success"] / (df_new["previous"] + 1)
        month_order = {'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6, 'jul' : 7,
                       'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12}
        df_new["month_num"] = df_new["month"].map(month_order).fillna(0)
        day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
        df_new["day_num"] = df_new["day_of_week"].map(day_map).fillna(0)


        logger.info("Advanced feature engineering complete.")
        return df_new
    


    def get_native_catboost_preprocessor(self):
        ALL_NUMERIC_COLS = [
            'age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
            'was_contacted_prev', 'poutcome_success', 'age_sq',
            'aggression_ratio', 'economic_mood', 'young_student', 'senior_retired',
            'success_rate_prev', 'month_num', 'day_num'
        ]
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        return num_pipeline, ALL_NUMERIC_COLS
    
    def initiate_data_transformation(self):
        try:
            logger.info("Starting data transformation process...")
            train_df_raw = pd.read_csv(self.data_ingestion_config.train_data_path)
            test_df_raw = pd.read_csv(self.data_ingestion_config.test_data_path)
            logger.info("Applying feature engineering to train and test sets.")
            
            # Apply feature engineering
            train_df = self._feature_engineer(train_df_raw)
            test_df = self._feature_engineer(test_df_raw)

            num_pipeline, ALL_NUMERIC_COLS = self.get_native_catboost_preprocessor()
            logger.info("Fitting numeric preprocessor on training data...")
            # Transform numeric columns
            train_df[ALL_NUMERIC_COLS] = num_pipeline.fit_transform(train_df[ALL_NUMERIC_COLS])
            test_df[ALL_NUMERIC_COLS] = num_pipeline.transform(test_df[ALL_NUMERIC_COLS])
            # Fill missing values in categorical columns with 'missing'
            for col in train_df.select_dtypes(include=['object']).columns:
                train_df[col] = train_df[col].fillna('missing')
                test_df[col] = test_df[col].fillna('missing')
            # Save the preprocessor object
            save_object(
                file_path=self.scalar_path,
                obj=num_pipeline
            )    
            # Save transformed datasets
            logger.info(f"Saved numeric preprocessor object to {self.scalar_path}")
            train_df.to_csv(Path(self.config.engineered_train_data_path), index=False)
            test_df.to_csv(Path(self.config.engineered_test_data_path), index=False)
            logger.info(f"Saved engineered train data to {self.config.engineered_train_data_path}")
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise e
        