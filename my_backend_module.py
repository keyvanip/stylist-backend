import os
import json
import numpy as np
import requests
import random
from typing import List, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime, timedelta
import faiss
import re

# Load environment variables
load_dotenv()

# Initialize Anthropic Client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Weather Tool
class WeatherTool:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEATHERMAP_API_KEY not found in environment variables.")

    def get_temperature(self, city="Washington", country=None, forecast_day: int = 0) -> Tuple[str, str, str]:
        if country:
            query = f"{city},{country}"
        else:
            query = city

        if forecast_day == 0:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={query}&appid={self.api_key}&units=metric"
        else:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={query}&appid={self.api_key}&units=metric"

        response = requests.get(url)
        data = response.json()

        if "cod" in data and data["cod"] not in ["200", 200]:
            raise ValueError(f"City '{city}' not found. OpenWeatherMap Error: {data.get('message', '')}")

        if forecast_day == 0:
            temp_c = round(data["main"]["temp"])
            temp_f = round((temp_c * 9 / 5) + 32)
            date_info = datetime.now().strftime("%A, %B %d")
            return f"{temp_c}°C / {temp_f}°F", city, date_info
        else:
            forecast_list = data.get("list", [])
            forecast_time = datetime.now() + timedelta(days=forecast_day)
            target_time_str = forecast_time.strftime("%Y-%m-%d 12:00:00")
            for entry in forecast_list:
                if entry["dt_txt"] == target_time_str:
                    temp_c = round(entry["main"]["temp"])
                    temp_f = round((temp_c * 9 / 5) + 32)
                    date_info = forecast_time.strftime("%A, %B %d")
                    return f"{temp_c}°C / {temp_f}°F", city, date_info

        raise ValueError(f"Failed to fetch weather data: {data}")

# Simple city extractor from user message
def extract_city_from_message(message: str) -> str:
    city_candidates = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', message)
    if city_candidates:
        return city_candidates[-1]
    return "Washington"

# Occasion Parser
def parse_occasion(user_input):
    user_input = user_input.lower()
    if any(word in user_input for word in ["date", "romantic"]): return "date night"
    if any(word in user_input for word in ["interview", "meeting"]): return "job interview"
    if any(word in user_input for word in ["beach", "walk"]): return "beach walk"
    if any(word in user_input for word in ["chill", "weekend", "hang"]): return "chill weekend hang"
    if any(word in user_input for word in ["formal"]): return "formal"
    if any(word in user_input for word in ["gym", "run", "sporty"]): return "sporty"
    return "casual"

# Preference Parser
def parse_clothing_preferences(user_input):
    user_input = user_input.lower()
    preferred_tops = []
    preferred_bottoms = []
    excluded_tops = []
    excluded_bottoms = []

    t_shirt_keywords = ["t-shirt", "t shirt", "tshirts", "t-shirts"]
    shirt_keywords = ["shirt", "shirts"]

    mentions_tshirt = any(keyword in user_input for keyword in t_shirt_keywords)
    mentions_shirt = any(keyword in user_input for keyword in shirt_keywords) and not mentions_tshirt

    if mentions_tshirt and not mentions_shirt:
        preferred_tops.append("t-shirt")
        excluded_tops.append("shirt")
    elif mentions_shirt and not mentions_tshirt:
        preferred_tops.append("shirt")
        excluded_tops.append("t-shirt")

    if "shorts" in user_input and "pants" not in user_input:
        preferred_bottoms.append("shorts")
        excluded_bottoms.append("pants")
    if "pants" in user_input and "shorts" not in user_input:
        preferred_bottoms.append("pants")
        excluded_bottoms.append("shorts")

    return preferred_tops, preferred_bottoms, excluded_tops, excluded_bottoms

# Load metadata dynamically
def load_user_metadata(user):
    path = f"data/{user.lower()}_metadata.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metadata found for user: {user}")
    with open(path) as f:
        return json.load(f)

# Wardrobe Filter
def filter_wardrobe(wardrobe_dict, temperature, occasion, color_preference=None, user_input=""):
    preferred_tops, preferred_bottoms, excluded_tops, excluded_bottoms = parse_clothing_preferences(user_input)

    suitable = []
    for items in wardrobe_dict.values():
        for item in items:
            item_styles = item.get("style", [])
            if isinstance(item_styles, str):
                item_styles = [item_styles]

            # 🔥 Correct strict filtering
            if occasion in item_styles:
                if temperature < 15:
                    if item["category"] == "pants" or item.get("sleeve", "") == "long":
                        suitable.append(item)
                else:
                    suitable.append(item)

    # Safety net if too few items
    if len(suitable) < 2:
        for items in wardrobe_dict.values():
            for item in items:
                styles = item.get("style", [])
                if isinstance(styles, str):
                    styles = [styles]
                if occasion in styles or (occasion != "formal" and "casual" in styles):
                    suitable.append(item)

    # Apply color preference (if any)
    if color_preference:
        if color_preference in ["dark", "bright"]:
            suitable = [item for item in suitable if color_preference in item.get("color", "").lower()] or suitable
        else:
            suitable = [item for item in suitable if color_preference.lower() in item.get("color", "").lower()] or suitable

    # Apply category preferences (if any)
    if preferred_tops:
        suitable = [item for item in suitable if item["category"] not in excluded_tops or item["category"] in preferred_tops]
    if preferred_bottoms:
        suitable = [item for item in suitable if item["category"] not in excluded_bottoms or item["category"] in preferred_bottoms]

    return suitable

# Generate Recommendations
def generate_outfit_recommendations(user_input, temperature, gender, outfits: List[List[dict]]):
    prompt = f"""
Hey, you're my stylish buddy helping me get dressed.
I'm a {gender} and I'm trying to figure out what to wear.

Here’s the situation:
"{user_input}" (temperature is {temperature})

Here are three outfit combos I’m considering:
"""

    for i, outfit in enumerate(outfits):
        prompt += f"\nOutfit {i+1}:\n{json.dumps(outfit, indent=2)}\n"

    prompt += """
For each one, tell me what kind of vibe it gives off — like if I want to feel chill, sporty, bold, etc.
Then give me your final verdict on which one to wear and why. Keep it casual and short, like we're texting.

Format your response as a JSON like this:
{
  "outfit_descriptions": [
    "Short 2-line vibe for outfit 1...",
    "Short 2-line vibe for outfit 2...",
    "Short 2-line vibe for outfit 3..."
  ],
  "final_recommendation": "Final casual-text verdict here."
}
Only return valid JSON. No markdown or explanation.
"""

    completion = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=400,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = completion.content[0].text.strip()
    try:
        parsed = json.loads(response_text)
        return parsed["outfit_descriptions"], parsed["final_recommendation"]
    except Exception as e:
        print("⚠️ Failed to parse response:", response_text)
        raise e

# Outfit Agent
def outfit_agent(user_input, temperature, gender="male", refresh=False, lock_top=None, lock_bottom=None, color=None, user="parsa"):
    wardrobe = load_user_metadata(user)
    occasion = parse_occasion(user_input)

    temp_str, location, forecast_date = temperature
    numeric_temp = int(temp_str.split("°C")[0])

    print(f"\n👤 Wardrobe selected: {user}")
    print(f"🌤️ Detected temperature in {location} on {forecast_date}: {temp_str}")
    print(f"🎯 Occasion detected: {occasion}\n")

    filtered = filter_wardrobe(wardrobe, numeric_temp, occasion, color, user_input=user_input)
    tops = [item for item in filtered if item["category"] in ["shirt", "t-shirt"]]
    bottoms = [item for item in filtered if item["category"] in ["pants", "shorts"]]

    if refresh:
        random.shuffle(tops)
        random.shuffle(bottoms)

    combinations = []
    for i in range(min(3, len(tops), len(bottoms))):
        top = lock_top if lock_top else tops[i]
        bottom = lock_bottom if lock_bottom else bottoms[i]
        combinations.append([top, bottom])

    descriptions, final_note = generate_outfit_recommendations(user_input, temp_str, gender, combinations)

    return combinations, descriptions, final_note

# Exported symbols
__all__ = [
    "WeatherTool",
    "outfit_agent",
    "generate_outfit_recommendations",
    "parse_occasion",
    "extract_city_from_message"
]