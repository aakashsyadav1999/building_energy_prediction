import pandas as pd
import pickle
from dataclasses import dataclass,field
import os

from src.constants import *
from src.entity.config_entity import ModelTrainerConfig,DataTransformationConfig
from src.logger import logging

from imblearn.over_sampling import SMOTE

#Defining models for prediction
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.common import save_object


@dataclass
#class for initiating all methods
class ModelTrainer:

    def __init__ (self,model_trainer_config:ModelTrainerConfig):
    
        self.model_trainer_config = model_trainer_config
        self.trained_model_file_path=os.path.join('MODEL_DIR','model.pkl')


    def model_train(self,train_array,test_array):

        try:

            logging.info("Enter into Model Building")
            # Define features and target variables

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Initialize the model
            rf = RandomForestRegressor(random_state=42)

            # Perform grid search
            grid_search = GridSearchCV(estimator=rf, param_grid=self.model_trainer_config.params, cv=5, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            # Get the best parameters and model
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_

            # Make predictions on the cleaned test data
            y_pred_best = best_model.predict(X_test)

            # Evaluate the model's performance on the cleaned data
            mse_best = mean_squared_error(y_test, y_pred_best)
            mae_best = mean_absolute_error(y_test, y_pred_best)
            r2_best = r2_score(y_test, y_pred_best)

            print(f'Best Mean Squared Error (MSE): {mse_best:.2f}')
            print(f'Best Mean Absolute Error (MAE): {mae_best:.2f}')
            print(f'Best R^2 Score: {r2_best:.2f}')
            print(f'Best Parameters: {best_params}')
            # Creating directories if they don't exist
            #os.makedirs(self.data_transformation_config.model_dir, exist_ok=True)
            
            save_object(
                file_path=self.trained_model_file_path,
                obj=best_model
            )
        
        
        except Exception as e:
            raise e


    def initiate_model_trainer(self,train_arr,test_arr):

        logging.info("Entered the initiate_model_trainer method of the model trainer class")
        try:
            os.makedirs(
                self.model_trainer_config.model_trainer_dir,exist_ok=True
            )

            self.model_train(train_arr,test_arr)

        except Exception as e:
            raise e