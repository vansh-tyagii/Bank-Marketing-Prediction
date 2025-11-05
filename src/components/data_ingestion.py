# In src/components/data_ingestion.py
import requests
import zipfile
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path
from src.utils.logger import logger 

# tried my best to explain each step with logging
# hahh but honestly this is straightforward
# and i also learn lot like unzip this way using zipfile module
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        Downloads and extracts the zip file.
        """
        try:
            logger.info(f"Downloading data from {self.config.source_url}")
            r = requests.get(self.config.source_url)

            # Save the zip file
            with open(self.config.local_zip_file, 'wb') as f:
                f.write(r.content) 
            logger.info(f"Saved zip file to {self.config.local_zip_file}")

            # Unzip the file
            with zipfile.ZipFile(self.config.local_zip_file, 'r') as z:
                z.extractall(self.config.unzip_dir)
            logger.info(f"Unzipped file to {self.config.unzip_dir}")

        except Exception as e:
            logger.error(f"Error in downloading data: {e}")
            raise e

    def clean_and_split_data(self):
        """
        Reads the raw CSV, handles leakage, and splits the data.
        """
        try:
            logger.info("Reading and cleaning raw data...")
            df = pd.read_csv(self.config.raw_data_file, sep=';') # because the file uses ';' as separator

            # Drop true duplicate rows before dropping 'duration' column
            initial_rows = len(df)
            df.drop_duplicates(inplace=True)
            # reset index after dropping duplicates
            df.reset_index(drop=True, inplace=True)
            logger.info(f"shape: {df.shape}")
            final_rows = len(df)
            dropped_count = initial_rows - final_rows
            logger.info(f"Dropped {dropped_count} duplicate rows.")

            # this frustrating 'duration' column causes data leakage
            # because it directly correlates with the target variable 'y'
            df = df.drop('duration', axis=1)
            logger.info("Dropped 'duration' column to prevent data leakage.")

            # Convert target variable 'y' to binary or say target encoding
            df['y'] = df['y'].map({'yes': 1, 'no': 0})

            # split the data with stratification for imbalanced classes
            logger.info("Performing stratified train-test split.")
            train_df, test_df = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42, 
                stratify=df['y'] # target is imbalanced
            )
            logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}") 
            # check distribution of target variable stratification worked or not 
            logger.info(f"Positive class in train: {train_df['y'].mean()}, Positive class in test: {test_df['y'].mean()}")

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Saved train data to {self.config.train_data_path}")
            logger.info(f"Saved test data to {self.config.test_data_path}")

        except Exception as e:
            logger.error(f"Error in cleaning and splitting data: {e}")
            raise e

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion process...")
        self.download_data()
        self.clean_and_split_data()
        logger.info("Data ingestion process completed.")