import os
import sys
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from src.exception import NerException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_object
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.encoding_map_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        logging.info("Starting fit in TargetEncoder")
        if X is None or y is None:
            logging.error("Input to fit is None")
            raise ValueError("Input to fit should not be None")
        assert len(X) == len(y)
        
        self.encoding_map_ = {}
        y = y.reset_index(drop=True)
        
        # Calculate the global mean
        self.global_mean_ = np.mean(y)
        logging.info(f"Global mean: {self.global_mean_}")
        
        for col in X.columns:
            col_values = X[col]
            averages = y.groupby(col_values).mean()
            counts = y.groupby(col_values).count()
            
            smooth = (counts * averages + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.encoding_map_[col] = smooth
            logging.info(f"Encoding map for column {col}: {self.encoding_map_[col]}")
            
        return self

    def transform(self, X):
        logging.info("Starting transform in TargetEncoder")
        if X is None:
            logging.error("Input to transform is None")
            raise ValueError("Input to transform should not be None")
        
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.encoding_map_:
                X_transformed[col] = X[col].map(self.encoding_map_[col])
                X_transformed[col].fillna(self.global_mean_, inplace=True)
            else:
                logging.warning(f"Column {col} not found in encoding map. Filling with global mean.")
                X_transformed[col] = self.global_mean_
            
        return X_transformed

    def fit_transform(self, X, y=None):
        logging.info("Starting fit_transform in TargetEncoder")
        return self.fit(X, y).transform(X)


@dataclass
class EnsureDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X
    

@dataclass
class RemoveCommasTransformer(BaseEstimator, TransformerMixin):
    columns_to_remove_comma: list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns_to_remove_comma:
            X[col] = X[col].astype(str).str.replace(",", "").astype(float)
        return X


@dataclass
class DataTransformation:
    
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.preprocessor_obj_file = os.path.join('MODEL_DIR', 'preprocessor.pkl')

    def drop_columns(self, df, columns_to_drop):
        if not isinstance(df, pd.DataFrame):
            logging.error("drop_columns: Input should be a DataFrame")
            raise ValueError("Input should be a DataFrame")
        return df.drop(columns=columns_to_drop, axis=1)
    
    def drop_columns_function(self, df):
        return self.drop_columns(df, self.data_transformation_config.column_to_drop)


    def get_data_transformation_object(self):
        try:
            columns_to_drop = self.data_transformation_config.column_to_drop
            cat_cols = self.data_transformation_config.one_hot_encoding
            ordinal_encode_cols = self.data_transformation_config.ordinal_encoding
            target_encode_cols = self.data_transformation_config.target_encoding
            columns_to_remove_comma = self.data_transformation_config.comma_removal



            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
            ])

            ordinal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
            ])

            target_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ensure_dataframe", EnsureDataFrame()),
            ("target_encoder", TargetEncoder(smoothing=2.0))
            ])

            remove_commas_pipeline = Pipeline(steps=[
                ("remove_commas", RemoveCommasTransformer(columns_to_remove_comma))
            ])

            logging.info(f"Categorical Columns: {cat_cols}")
            logging.info(f"Ordinal Encode Columns: {ordinal_encode_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("remove_commas_pipeline", remove_commas_pipeline, columns_to_remove_comma),
                    ("cat_pipeline", cat_pipeline, cat_cols),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_encode_cols),
                    ("target_pipeline", target_pipeline, target_encode_cols),
                    ("drop_columns", FunctionTransformer(self.drop_columns_function), columns_to_drop),
                ],
                remainder='passthrough'
            )
            return preprocessor

        except Exception as e:
            raise NerException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading the train and test files")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if train_df is None or test_df is None:
                logging.error("Train or test dataframe is None")
                raise ValueError("Train or test dataframe is None")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = self.data_transformation_config.target_column

            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                logging.error(f"Target column '{target_column_name}' not found in train or test dataframe")
                raise ValueError(f"Target column '{target_column_name}' not found in train or test dataframe")

            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframes")

            if input_features_train_df.empty or target_feature_train_df.empty:
                logging.error("Training data is empty after dropping target column")
                raise ValueError("Training data is empty after dropping target column")

            if input_feature_test_df.empty or target_feature_test_df.empty:
                logging.error("Test data is empty after dropping target column")
                raise ValueError("Test data is empty after dropping target column")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file
            )

        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise NerException(e, sys)