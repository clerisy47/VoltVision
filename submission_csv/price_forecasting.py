import joblib
import pandas as pd
from datetime import datetime, timedelta

model = joblib.load('model/price_forecasting.pkl')
start = 34825
end = 34825+24*7-1    
predictions = pd.DataFrame(model.predict(start=start, end=end ))
start_date = datetime(2023, 12, 25)
start_hour = 1
date_hour_combinations = [(start_date + timedelta(days=d, hours=h)).strftime('%m/%d/%Y %H:%M') for d in range(7) for h in range(start_hour, start_hour + 24)]

predictions['Date'] = date_hour_combinations
predictions['Hour'] = [f'h{i}' for i in range(start_hour, start_hour + 24)] * 7

predictions.to_csv('predicted_prices.csv', index=False)