from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import re

# Load your backend logic
from my_backend_module import outfit_agent, WeatherTool, parse_occasion

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define expected input fields
class OutfitRequest(BaseModel):
    message: str
    gender: Optional[str] = "male"
    color: Optional[str] = None
    lock_top: Optional[dict] = None
    lock_bottom: Optional[dict] = None
    user: Optional[str] = "parsa"

# Initialize weather tool
weather_tool = WeatherTool()

@app.post("/generate-outfit")
def generate_outfit(req: OutfitRequest):
    # Determine city and day from user message
    lowered = req.message.lower()

    forecast_day = 1 if "tomorrow" in lowered else 0

    city_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', req.message)
    default_city = "Washington"

    if city_candidates:
        city = city_candidates[-1]
    else:
        city = default_city

    print(f"🌍 Detected city for weather lookup: {city}")

    temperature, detected_city, forecast_date = weather_tool.get_temperature(city=city, forecast_day=forecast_day)

    combos, outfit_descs, final_recommendation = outfit_agent(
        user_input=req.message,
        temperature=(temperature, detected_city, forecast_date),
        gender=req.gender,
        lock_top=req.lock_top,
        lock_bottom=req.lock_bottom,
        color=req.color,
        refresh=True,
        user=req.user
    )

    return {
        "outfits": combos,
        "outfitDescriptions": outfit_descs,
        "message": final_recommendation,
        "temperature": temperature,
        "location": detected_city,
        "forecastDate": forecast_date,
        "occasion": parse_occasion(req.message)
    }