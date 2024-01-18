import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model/price_forecasting.pkl')

def main():
    st.title("Price Forecasting App")

    date = st.date_input("Select a date", pd.to_datetime("today"))
    time = st.time_input("Select a time", pd.to_datetime("12:00 PM").time())
    datetime_input = pd.to_datetime(f"{date} {time}")

    prediction = predict_price(datetime_input)

    st.subheader("Price Prediction:")
    st.write(f"The predicted price at {datetime_input} is: {prediction:.2f} USD")

def predict_price(datetime_input):
    features = generate_features(datetime_input)
    prediction = model.predict(features.reshape(1, -1))[0]
    return prediction

def generate_features(datetime_input):
    features = np.random.rand(10)
    return features

if __name__ == "__main__":
    main()
