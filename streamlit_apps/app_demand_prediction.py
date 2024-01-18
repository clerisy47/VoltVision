import streamlit as st
from pydantic import BaseModel
import joblib

class WeatherData(BaseModel):
    temperature: float
    feelslike: float
    dewpoint: float
    humidity: float
    precipitation: float
    precipprob: float
    preciptype: str
    snow: float
    snowdepth: float
    windgust: float
    windspeed: float
    winddirection: float
    sealevelpressure: float
    cloudcover: float
    visibility: float
    solarradiation: float
    uvindex: float
    severerisk: float
    conditions: str

def predict_demand_price(weather_data: WeatherData, model):
    features = [weather_data.temperature, weather_data.feelslike, weather_data.dewpoint, weather_data.humidity,
                weather_data.precipitation, weather_data.precipprob, weather_data.snow, weather_data.snowdepth,
                weather_data.windgust, weather_data.windspeed, weather_data.winddirection,
                weather_data.sealevelpressure, weather_data.cloudcover, weather_data.visibility,
                weather_data.solarradiation, weather_data.uvindex, weather_data.severerisk]

    prediction = model.predict([features])[0]
    demand, price = prediction
    return {"demand": demand, "price": price}

def main():
    st.title("Demand and Price Prediction App")

    model = joblib.load('model/demand_forecasting.pkl')

    st.sidebar.header("Input Weather Data")
    weather_data = WeatherData(
        temperature=st.sidebar.number_input("Temperature", value=0.0),
        feelslike=st.sidebar.number_input("Feels Like", value=0.0),
        dewpoint=st.sidebar.number_input("Dewpoint", value=0.0),
        humidity=st.sidebar.number_input("Humidity", value=0.0),
        precipitation=st.sidebar.number_input("Precipitation", value=0.0),
        precipprob=st.sidebar.number_input("Precipitation Probability", value=0.0),
        snow=st.sidebar.number_input("Snow", value=0.0),
        snowdepth=st.sidebar.number_input("Snow Depth", value=0.0),
        windgust=st.sidebar.number_input("Wind Gust", value=0.0),
        windspeed=st.sidebar.number_input("Wind Speed", value=0.0),
        winddirection=st.sidebar.number_input("Wind Direction", value=0.0),
        sealevelpressure=st.sidebar.number_input("Sea Level Pressure", value=0.0),
        cloudcover=st.sidebar.number_input("Cloud Cover", value=0.0),
        visibility=st.sidebar.number_input("Visibility", value=0.0),
        solarradiation=st.sidebar.number_input("Solar Radiation", value=0.0),
        uvindex=st.sidebar.number_input("UV Index", value=0.0),
        severerisk=st.sidebar.number_input("Severe Risk", value=0.0),
        conditions=st.sidebar.text_input("Conditions", value=""),
    )

    if st.sidebar.button("Predict"):
        try:
            result = predict_demand_price(weather_data, model)
            st.success(f"Predicted Demand: {result['demand']}, Predicted Price: {result['price']}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
