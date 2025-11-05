from box import ConfigBox 
from pathlib import Path 
from src.utils.logger import logger
from box.exceptions import BoxValueError 
import os
import yaml
import joblib

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully.")
            return ConfigBox(content)  
        
    except BoxValueError:
        raise ValueError("YAML file is empty")    
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True) # create directory if it doesn't exist
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_object(file_path: Path, obj: object):
    """Saves a Python object to a file using joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved successfully to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")
        raise e

def load_object(file_path: Path) -> object:
    """Loads a Python object from a file using joblib."""
    try:
        obj = joblib.load(file_path)
        logger.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}")
        raise e            