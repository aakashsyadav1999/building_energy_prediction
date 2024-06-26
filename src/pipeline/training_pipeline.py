import sys
import time
import stat
from src.constants import *

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    DataIngestionConfig,DataTransformationConfig,ModelTrainerConfig
    
)


from src.exception import NerException
from src.logger import logging

#Class create to start all the process which is in components.
class TrainPipeline:
    
        def __init__(self):
            self.data_ingestion_config = DataIngestionConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()


        #Permission for directory to read, write, delete    
        def create_directory_with_permissions(self, directory_path):
            try:
                os.makedirs(directory_path, exist_ok=True)
                os.chmod(directory_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # Set permissions for owner, group, and others
                logging.info(f"Directory '{directory_path}' created successfully.")
            except Exception as e:
                raise NerException(e, sys) from e


        # This method is used to start the data ingestion
        def start_data_ingestion(self):
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            try:
                directory_path = self.data_ingestion_config.data_ingestion_artifacts_dir
                self.create_directory_with_permissions(directory_path)
                logging.info(f"Creating {directory_path}")
                logging.info("Getting the data from Google drive storage")
                data_ingestion = DataIngestion(
                    data_ingestion_config=self.data_ingestion_config
                )
                data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
                logging.info("Got the data from Google drive storage")
                logging.info(
                    "Exited the start_data_ingestion method of TrainPipeline class"
                )
                return data_ingestion_artifact

            except Exception as e:
                raise NerException(e, sys) from e
            

        def start_data_transformation(self) -> DataTransformationConfig:
            logging.info("Entered Data Transformation method in training pipeline")
            try:
                directory_path = self.data_transformation_config.data_transformation_dir
                self.create_directory_with_permissions(directory_path)
                logging.info(f"Creating {directory_path}")

                logging.info("Starting Data Transformation")
                #Make object of data transformation config and all the initate data transformation function.
                data_transformation = DataTransformation(
                    data_transformation_config=self.data_transformation_config
                )
                
                #call function
                train_path = r'D:\VS code files\building_energy_consumption\dataset\singapore_dataset\end_end\artifacts\DataTransformationArtifacts\train.csv'
                test_path = r'D:\VS code files\building_energy_consumption\dataset\singapore_dataset\end_end\artifacts\DataTransformationArtifacts\test.csv'
            
                train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_path,test_path)
                logging.info("Cleaned Data")
                logging.info("Exited the data_transformation method of TrainPipeline class")
                return train_arr,test_arr
            except Exception as e:
                pass
                raise NerException(e, sys) from e


        #This method starts model trainer.
        def start_model_trainer(self,train_arr, test_arr) -> ModelTrainerConfig:
            logging.info("Entered Model Building in training pipeline")
            try:
                directory_path = self.model_trainer_config.model_trainer_dir
                self.create_directory_with_permissions(directory_path)
                logging.info(f"Creating {directory_path}")

                logging.info("Starting Model Trainer")
                #Make object of model transformation config and all the initate model transformation function.
                model_trainer = ModelTrainer(
                    model_trainer_config=self.model_trainer_config
                )
                #call function
                model_trainer= model_trainer.initiate_model_trainer(train_arr, test_arr)
                logging.info("Creating model")
                logging.info("Exited the model trainer method of TrainPipeline class")
                return model_trainer 
            except Exception as e:
                raise NerException(e, sys) from e



        # This method is used to start the training pipeline
        def run_pipeline(self) -> None:
            try:
                logging.info("Started Model training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                data_ingestion_artifact = self.start_data_ingestion()
                train_arr,test_arr = self.start_data_transformation()
                model_transformations = self.start_model_trainer(train_arr,test_arr)
            except Exception as e:
                raise e