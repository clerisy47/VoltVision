# import joblib
# import pandas as pd
# from data_preprocessing import dataPreprocessing
# from feature_engineering import featureEngineering

# df_submit = pd.read_csv('../../dataset/Demand Forecasting/Demand Forecasting Weather Data upto Feb 28.csv', sep=',')[27554:]
# df_submit = dataPreprocessing(df=df_submit)
# df_submit = featureEngineering(df=df_submit)
# print(df_submit.head())

import os

file_path = '../../dataset/Demand Forecasting/Demand Forecasting Weather Data upto Feb 28.csv'
print(os.path.exists(file_path))