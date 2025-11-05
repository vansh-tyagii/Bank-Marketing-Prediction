from pydantic import BaseModel, DirectoryPath, FilePath, AnyUrl
from pathlib import Path

# learn pydantic so why not use it for config entities
class DataIngestionConfig(BaseModel):
    root_dir: DirectoryPath
    source_url: AnyUrl
    local_zip_file: Path
    unzip_dir: DirectoryPath
    raw_data_file: Path
    train_data_path: Path 
    test_data_path: Path 

class DataTransformationConfig(BaseModel):
    root_dir: DirectoryPath
    preprocessor_obj_file: Path
    engineered_train_data_path: Path
    engineered_test_data_path: Path

class ModelTrainerConfig(BaseModel):
    root_dir: DirectoryPath
    trained_model_file_path: Path
    train_data_path: Path
    test_data_path: Path    