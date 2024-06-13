
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
import template
from flask_cors import CORS
application=Flask(__name__)

app=application
CORS = app
## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            
            Building_Type=str(request.form.get('Building_Type', '')),
            Main_Function=str(request.form.get('Main_Function', '')),
            Building_Size=str(request.form.get('Building_Size', '')),
            EUI2020=int(request.form.get('EUI2020', '0') or '0'),
            EUI_Quartile__Energy_Ranking2020=str(request.form.get('EUI_Quartile__Energy_Ranking2020', '')),
            EUI2021_=int(request.form.get('EUI2021_', '0') or '0'),
            EUI_Quartile__Energy_Ranking_2021=str(request.form.get('EUI_Quartile__Energy_Ranking_2021', '')),
            EUI_Quartile__Energy_Ranking_2022=str(request.form.get('EUI_Quartile__Energy_Ranking_2022', '')),
            TOP_CSC_Year=int(request.form.get('TOP_CSC_Year', '0') or '0'),
            Award__Green_Non_Green_=str(request.form.get('Award__Green_Non_Green_', '')),
            Green_Mark_Version=str(request.form.get('Green_Mark_Version', '')),
            GFA=int(request.form.get('GFA', '0') or '0'),
            AC_Area=int(request.form.get('AC_Area', '0') or '0'),
            AC_Area_Percentage=str(request.form.get('AC_Area_Percentage', '')),
            No_Of_Hotel_Room=int(request.form.get('No_Of_Hotel_Room', '0') or '0'),
            AC_Type=str(request.form.get('AC_Type', '')),
            Age__of_Chiller=int(request.form.get('Age__of_Chiller', '0') or '0'),
            Aircon_system_efficiency__kW_RT_=float(request.form.get('Aircon_system_efficiency__kW_RT_', '0.0') or '0.0')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
