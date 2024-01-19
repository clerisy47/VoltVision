import joblib
import pandas as pd
from steps.demand_forecasting.data_preprocessing import dataPreprocessing
from steps.demand_forecasting.feature_engineering import featureEngineering

df_submit = pd.read_csv('../dataset/Demand Forecasting/Demand Forecasting Weather Data upto Feb 28.csv', sep=',')[27554:]
df_submit = dataPreprocessing(df_submit)
df_submit = featureEngineering(df_submit)
model = joblib.load('../models/demand_forecasting.pkl')
df_submit['Demand (MW)'] = model.predict(df_submit)
df_sumnit = df_submit[['datetime', 'Demand (MW)']]