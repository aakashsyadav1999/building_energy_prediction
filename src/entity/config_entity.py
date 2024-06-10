from dataclasses import dataclass
import os
from pathlib import Path
from src.constants import *
from src.entity.config_entity import *


#Data Ingestion
@dataclass
class DataIngestionConfig:

    def __init__ (self):

        self.data_ingestion_artifacts_dir:str = os.path.join(
            ARTIFACTS_DIR,DATA_INGESTION_ROOT
        )

        self.source_url:str = SOURCE_URL

        self.local_data_file:str = os.path.join(
            self.data_ingestion_artifacts_dir,LOCAL_FILE_PATH
        )

        self.unzip_dir:str = UNZIP_DIR

        self.unzip_csv_data:str = os.path.join(
            ARTIFACTS_DIR,UNZIP_DIR_CSV_DATA
        )

        self.split_train_test:str = os.path.join(
            ARTIFACTS_DIR,SPLIT_TRAIN_TES_DATA
        )

#Data Transformation
@dataclass
class DataTransformationConfig:

    def __init__ (self):

        self.data_transformation_dir:str = os.path.join(
            ARTIFACTS_DIR,DATA_TRANSFORMATION_DIR
        )
        self.data_transformation_file_name: str = DATA_TRANSFORMATION_FILE

        self.column_to_drop:list = COLUMNS_TO_DROP

        self.ordinal_encoding:list = ORDINAL_ENCODING

        self.one_hot_encoding:list = ONE_HOT_ENCODING

        self.target_encoding:list = TARGET_ENCODING

        self.target_column:list = TARGET_COLUMN