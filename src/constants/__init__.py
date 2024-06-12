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
                    '2020_EUI',
                    '2020_EUI_Quartile/_Energy_Ranking',
                    '2021_EUI',
                    '2021_EUI_Quartile/_Energy_Ranking',
                    '2022_EUI_Quartile/_Energy_Ranking',
                    'AC_Area_Percentage'
                ]

ORDINAL_ENCODING = [
                    
                    'Award_(Green/Non-Green)'
                
                ]

ONE_HOT_ENCODING = [
                    
                    'Building_Type', 
                    'Main_Function',
                    'Building_Size'

                ]

TARGET_ENCODING = [
                    'Green_Mark_Version',
                    'AC_Type',
                    'TOP/CSC_Year'
                ]

TARGET_COLUMN = '2022_EUI'



COMMA_REMOVAL = [
                    'GFA',
                    'AC_Area'
                ]


#Model Building
MODEL_TRAINING_ARTIFACTS_DIR = "ModelTrainingArtifacts"

# Define the parameter grid
PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}



