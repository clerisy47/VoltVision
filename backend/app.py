from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

model = joblib.load('your_model_path.pkl')

def predict_demand_price(weather_data: WeatherData):
    features = [weather_data.temperature, weather_data.feelslike, weather_data.dewpoint, weather_data.humidity,
                weather_data.precipitation, weather_data.precipprob, weather_data.snow, weather_data.snowdepth,
                weather_data.windgust, weather_data.windspeed, weather_data.winddirection,
                weather_data.sealevelpressure, weather_data.cloudcover, weather_data.visibility,
                weather_data.solarradiation, weather_data.uvindex, weather_data.severerisk]

    prediction = model.predict([features])[0]
    demand, price = prediction
    return {"demand": demand, "price": price}

@app.post("/predict")
async def get_prediction(weather_data: WeatherData):
    try:
        result = predict_demand_price(weather_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
