import pandas as pd
import os
from src.utils.common import save_object, load_object


data = {
    'Building Type': ['Commercial Building', 'Commercial Building', 'Commercial Building', 'Commercial Building'],
    'Main Function': ['Industrial', 'Industrial', 'Industrial', 'Office'],
    'Building Size': ['Large', 'Large', 'Large', 'Large'],
    '2020 EUI': [3725, 2044, 1157, 553],
    '2020 EUI Quartile/ Energy Ranking': ['N/A', 'Bottom Quartile', 'Bottom Quartile', 'Bottom Quartile'],
    '2021 EUI': [4453, 2214, 1182, 661],
    '2021 EUI Quartile/ Energy Ranking': ['N/A', 'Bottom Quartile', 'Bottom Quartile', 'Bottom Quartile'],
    #'2022 EUI': [4580, 2305, 1151, 805],
    '2022 EUI Quartile/ Energy Ranking': ['N/A', 'Bottom Quartile', 'Bottom Quartile', 'Bottom Quartile'],
    'TOP/CSC Year': [2015, 1999, 2014, 1986],
    'Award (Green/Non-Green)': ['Platinum', 'N/A', 'Platinum', 'N/A'],
    'Green Mark Version': ['Existing Data Centres', 'N/A', 'New Building for Non-Residential buildings (version 3)', 'N/A'],
    'GFA': ['35,218', '38,349', '19,173', '15,819'],
    'AC Area': ['28,536', '20,000', '19,173', '12,724'],
    'AC Area Percentage': ['81%', '52%', '100%', '80%'],
    'No. Of Hotel Room': [0, 0, 0, 0],
    'AC Type': ['Water Cooled Chilled Water Plant', 'Water Cooled Chilled Water Plant', 'Water Cooled Chilled Water Plant', 'Water Cooled Chilled Water Plant'],
    'Age  of Chiller': [7, 20, 5, 0],
    'Air-con system efficiency (kW/RT)': [0.744, 0, 0.578, 0]
}

sample_df = pd.DataFrame(data)

import pickle

model_path=os.path.join("MODEL_DIR","model.pkl")
preprocessor_path=os.path.join("MODEL_DIR","preprocessor.pkl")
print("Before Loading")

model = load_object(file_path=model_path)
preprocessor = load_object(file_path=preprocessor_path)


# Transform the sample data using the preprocessor
preprocessed_sample_data = preprocessor.transform(sample_df)

preprocessed_sample_data_df = pd.DataFrame(preprocessed_sample_data)
preprocessed_sample_data_df.to_csv("preprocessed_sample_data_df.csv")


# Predict using the loaded model
predictions = model.predict(preprocessed_sample_data_df)

# Combine predictions with the sample data
sample_df['Predictions'] = predictions

# Save the sample data with predictions to a CSV file
sample_df.to_csv('sample_data_with_predictions.csv', index=False)

print(sample_df)
print(preprocessed_sample_data)