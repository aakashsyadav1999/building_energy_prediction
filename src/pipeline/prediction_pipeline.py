import os
import sys
import pickle
import pandas as pd
from src.exception import NerException
from src.logger import logging
from src.utils.common import save_object,load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("MODEL_DIR","model.pkl")
            preprocessor_path=os.path.join("MODEL_DIR","preprocessor.pkl")
            print("Before Loading")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")
            print(f"Model type: {type(model)}")
            print(f"Preprocessor type: {type(preprocessor)}")

            if not hasattr(model, 'predict'):
                raise TypeError("Loaded model object does not have a 'predict' method")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        
        except Exception as e:
            raise NerException(e,sys)
        
class CustomData:

    def __init__ (self,
                  Building_Type: str,
                  Main_Function: str,
                  Building_Size: str,
                  EUI2020: int,
                  EUI_Quartile__Energy_Ranking2020: str,
                  EUI2021_: int,
                  EUI_Quartile__Energy_Ranking_2021: str,
                  EUI_Quartile__Energy_Ranking_2022: str,
                  TOP_CSC_Year: int,
                  Award__Green_Non_Green_:str,
                  Green_Mark_Version:str,
                  GFA: int,
                  AC_Area: int,
                  AC_Area_Percentage: str,
                  No_Of_Hotel_Room: int,
                  AC_Type: str,
                  Age__of_Chiller: int,
                  Aircon_system_efficiency__kW_RT_: int
                  ):
        

                self.Building_Type = Building_Type
                self.Main_Function = Main_Function
                self.Building_Size = Building_Size
                self.EUI2020 = EUI2020
                self.EUI_Quartile__Energy_Ranking2020 = EUI_Quartile__Energy_Ranking2020
                self.EUI2021_ = EUI2021_
                self.EUI_Quartile__Energy_Ranking_2021 = EUI_Quartile__Energy_Ranking_2021
                self.EUI_Quartile__Energy_Ranking_2022 = EUI_Quartile__Energy_Ranking_2022
                self.TOP_CSC_Year = TOP_CSC_Year
                self.Award__Green_Non_Green_ = Award__Green_Non_Green_
                self.Green_Mark_Version = Green_Mark_Version
                self.GFA = GFA
                self.AC_Area = AC_Area
                self.AC_Area_Percentage = AC_Area_Percentage
                self.No_Of_Hotel_Room = No_Of_Hotel_Room
                self.AC_Type = AC_Type
                self.Age__of_Chiller = Age__of_Chiller
                self.Aircon_system_efficiency__kW_RT_ = Aircon_system_efficiency__kW_RT_
        

    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                
                "Building_Type" : [self.Building_Type],
                "Main_Function" : [self.Main_Function],
                "Building_Size" : [self.Building_Size],
                "EUI2020" : [self.EUI2020],
                "EUI_Quartile__Energy_Ranking2020" : [self.EUI_Quartile__Energy_Ranking2020],
                "EUI2021_" : [self.EUI2021_],
                "EUI_Quartile__Energy_Ranking_2021" : [self.EUI_Quartile__Energy_Ranking_2021],
                "EUI_Quartile__Energy_Ranking_2022" : [self.EUI_Quartile__Energy_Ranking_2022],
                "TOP_CSC_Year" : [self.TOP_CSC_Year],
                "Award__Green_Non_Green_" : [self.Award__Green_Non_Green_],
                "Green_Mark_Version" : [self.Green_Mark_Version],
                "GFA" : [self.GFA],
                "AC_Area" : [self.AC_Area],
                "AC_Area_Percentage" : [self.AC_Area_Percentage],
                "No_Of_Hotel_Room" : [self.No_Of_Hotel_Room],
                "AC_Type" : [self.AC_Type],
                "Age__of_Chiller" : [self.Age__of_Chiller],
                "Aircon_system_efficiency__kW_RT_" : [self.Aircon_system_efficiency__kW_RT_]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise e