import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts")
LOGS_DIR = "logs"
LOGS_FILE_NAME = "SIDFC.log"

#Data Ingestion
DATA_INGESTION_ROOT = "DataIngestionArtifacts"
SOURCE_URL= "https://drive.google.com/file/d/1-uFAXfd4xUDJF7VdIkRwUebfYns9CnQ_/view?usp=sharing"
UNZIP_DIR= DATA_INGESTION_ROOT
LOCAL_FILE_PATH = "building_energy.zip"
UNZIP_DIR_CSV_DATA = "DataIngestionArtifacts"
SPLIT_TRAIN_TES_DATA = "DataTransformationArtifacts"

#Data transformation
DATA_TRANSFORMATION_DIR = "DataTransformationArtifacts"
DATA_TRANSFORMATION_FILE = 'cleaned_data.csv'
COLUMNS_TO_DROP = [
                    '2020 EUI',
                    '2020 EUI Quartile/ Energy Ranking',
                    '2021 EUI',
                    '2021 EUI Quartile/ Energy Ranking',
                    '2022 EUI Quartile/ Energy Ranking',
                    'AC Area Percentage'
                ]

ORDINAL_ENCODING = [
                    
                    'Award (Green/Non-Green)'
                
                ]

ONE_HOT_ENCODING = [
                    
                    'Building Type', 
                    'Main Function',
                    'Building Size'

                ]

TARGET_ENCODING = [
                    'Green Mark Version',
                    'AC Type'
                ]

TARGET_COLUMN = '2022 EUI'



