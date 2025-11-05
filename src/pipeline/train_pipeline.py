from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logger

class TrainPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_data_ingestion(self):
        logger.info("Starting Data Ingestion stage...")
        try:
            ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=ingestion_config)
            data_ingestion.initiate_data_ingestion()
            logger.info("Data Ingestion stage completed successfully.")
            return ingestion_config # Return config for the next step
        except Exception as e:
            logger.error(f"Data Ingestion stage failed: {e}")
            raise e

    def run_data_transformation(self, data_ingestion_config): # Takes config as input
        logger.info("Starting Data Transformation stage...")
        try:
            transform_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(
                config=transform_config,
                data_ingestion_config=data_ingestion_config
            )
            data_transformation.initiate_data_transformation()
            logger.info("Data Transformation stage completed successfully.")
        except Exception as e:
            logger.error(f"Data Transformation stage failed: {e}")
            raise e

    def run_model_training(self):
        logger.info("Starting Model Training stage...")
        try:
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.initiate_model_training()
            logger.info("Model Training stage completed successfully.")
        except Exception as e:
            logger.error(f"Model Training stage failed: {e}")
            raise e
        

if __name__ == "__main__":
    pipeline = TrainPipeline()
    ingestion_config = pipeline.run_data_ingestion()
    pipeline.run_data_transformation(ingestion_config)
    pipeline.run_model_training()