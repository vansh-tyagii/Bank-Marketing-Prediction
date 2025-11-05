# In src/components/model_trainer.py
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from src.entity.config_entity import ModelTrainerConfig
from src.utils.common import save_object
from src.utils.logger import logger
import warnings

# tried logistic xgboost and sequential but catboost is best
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _load_data(self):
        """Loads processed training and testing data."""

        logger.info("Loading processed data...")

        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        logger.info("Data loading complete.")
        logger.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
        
        # The last column is 'y (target)', everything else is 'X (features)'
        X_train = train_df.drop(columns=['y'])
        y_train = train_df['y']
        
        X_test = test_df.drop(columns=['y'])
        y_test = test_df['y']
        
        return X_train, y_train, X_test, y_test

    def initiate_model_training(self):
        warnings.filterwarnings("ignore", category=UserWarning) # Hide sklearn warnings
        logger.info("Starting model training process...")
        
        X_train, y_train, X_test, y_test = self._load_data()
       # These are the *best* params we found in our notebook
        champion_params = {
            'iterations': 1500, # Using the tuning loop's iterations
            'depth': 4,
            'learning_rate': 0.08,
            'l2_leaf_reg': 5,
            'eval_metric': 'PRAUC',
            'auto_class_weights': 'Balanced',
            'random_seed': 42,
            'verbose': 200,
            'early_stopping_rounds': 100
        }

        champion_model = CatBoostClassifier(**champion_params)

        cat_feature = X_train.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Categorical features identified: {cat_feature}, length: {len(cat_feature)}   ")
        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feature)
        test_pool = Pool(data=X_test, label=y_test, cat_features=cat_feature)

        champion_model.fit(train_pool, eval_set=test_pool)
        best_iteration = champion_model.get_best_iteration()
        logger.info(f"Champion model trained. Best iteration: {best_iteration}")

        y_pred_proba = champion_model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_pred_proba)  
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, champion_model.predict(X_test))
        logger.info(f"Champion Model PR-AUC: {pr_auc:.4f}")
        logger.info(f"Champion Model ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Classification Report:\n{report}")
        # saving model
        save_object(
            file_path=self.config.trained_model_file_path,
            obj=champion_model
        )

        logger.info(f"Saved champion model to {self.config.trained_model_file_path}")
        
       