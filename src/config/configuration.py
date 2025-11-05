from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig , DataTransformationConfig, ModelTrainerConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self, config_filepath="src/config/config.yaml"):
        self.config = read_yaml(Path(config_filepath))
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config.data_ingestion
        create_directories([ingestion_config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(ingestion_config.root_dir),
            source_url=ingestion_config.source_url,
            local_zip_file=Path(ingestion_config.local_zip_file),
            unzip_dir=Path(ingestion_config.unzip_dir),
            raw_data_file=Path(ingestion_config.raw_data_file),
            train_data_path=Path(ingestion_config.train_data_path),
            test_data_path=Path(ingestion_config.test_data_path)
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.config.data_transformation
        create_directories([transformation_config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(transformation_config.root_dir),
            # preprocessor_obj_file=Path(transformation_config.preprocessor_obj_file)
            preprocessor_obj_file=Path(transformation_config.preprocessor_obj_file),
            engineered_train_data_path=Path(transformation_config.engineered_train_data_path),
            engineered_test_data_path=Path(transformation_config.engineered_test_data_path)
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer_config = self.config.model_trainer
        transform_config = self.config.data_transformation
        create_directories([trainer_config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(trainer_config.root_dir),
            trained_model_file_path=Path(trainer_config.trained_model_file_path),
            train_data_path=Path(transform_config.engineered_train_data_path),
            test_data_path=Path(transform_config.engineered_test_data_path)
        )
        
        return model_trainer_config