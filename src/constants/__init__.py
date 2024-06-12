import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts")
LOGS_DIR = "logs"
LOGS_FILE_NAME = "SIDFC.log"

#Data Ingestion
DATA_INGESTION_ROOT = "DataIngestionArtifacts"
SOURCE_URL= "https://drive.google.com/file/d/1vIM4iYUO0INrK-iVUgPXABrwo8BZeDEY/view?usp=sharing"
UNZIP_DIR= DATA_INGESTION_ROOT
LOCAL_FILE_PATH = "building_energy.zip"
UNZIP_DIR_CSV_DATA = "DataIngestionArtifacts"
SPLIT_TRAIN_TES_DATA = "DataTransformationArtifacts"

#Data transformation
DATA_TRANSFORMATION_DIR = "DataTransformationArtifacts"
DATA_TRANSFORMATION_FILE = 'cleaned_data.csv'
COLUMNS_TO_DROP = [
                    'EUI2020',
                    'EUI_Quartile__Energy_Ranking2020',
                    'EUI2021_',
                    'EUI_Quartile__Energy_Ranking_2021',
                    'EUI_Quartile__Energy_Ranking_2022',
                    'AC_Area_Percentage'
                ]

ORDINAL_ENCODING = [
                    
                    'Award__Green_Non_Green_'
                
                ]

ONE_HOT_ENCODING = [
                    
                    'Building_Type', 
                    'Main_Function',
                    'Building_Size'

                ]

TARGET_ENCODING = [
                    'Green_Mark_Version',
                    'AC_Type',
                    'TOP_CSC_Year'
                ]

TARGET_COLUMN = 'EUI2022_'



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



