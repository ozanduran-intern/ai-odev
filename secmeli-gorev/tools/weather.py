"""
날씨 조회 도구
전 세계 주요 도시의 현재 날씨 정보를 시뮬레이션하여 제공
"""

from langchain_core.tools import tool


# 시뮬레이션된 날씨 데이터 (도시명 소문자 키 -> 날씨 정보)
WEATHER_DATA: dict[str, dict] = {
    # 한국
    "seoul": {"city": "Seoul (서울)", "temp": 12, "condition": "Clear / 맑음", "humidity": 45},
    "서울": {"city": "Seoul (서울)", "temp": 12, "condition": "Clear / 맑음", "humidity": 45},
    "busan": {"city": "Busan (부산)", "temp": 16, "condition": "Cloudy / 흐림", "humidity": 62},
    "부산": {"city": "Busan (부산)", "temp": 16, "condition": "Cloudy / 흐림", "humidity": 62},
    "jeju": {"city": "Jeju (제주)", "temp": 18, "condition": "Rainy / 비", "humidity": 78},
    "제주": {"city": "Jeju (제주)", "temp": 18, "condition": "Rainy / 비", "humidity": 78},
    "incheon": {"city": "Incheon (인천)", "temp": 11, "condition": "Partly Cloudy", "humidity": 50},
    "인천": {"city": "Incheon (인천)", "temp": 11, "condition": "Partly Cloudy", "humidity": 50},
    "daegu": {"city": "Daegu (대구)", "temp": 15, "condition": "Clear / 맑음", "humidity": 38},
    "대구": {"city": "Daegu (대구)", "temp": 15, "condition": "Clear / 맑음", "humidity": 38},
    # 터키
    "istanbul": {"city": "Istanbul", "temp": 14, "condition": "Cloudy", "humidity": 70},
    "이스탄불": {"city": "Istanbul", "temp": 14, "condition": "Cloudy", "humidity": 70},
    "ankara": {"city": "Ankara", "temp": 8, "condition": "Clear", "humidity": 35},
    "앙카라": {"city": "Ankara", "temp": 8, "condition": "Clear", "humidity": 35},
    "antalya": {"city": "Antalya", "temp": 22, "condition": "Sunny", "humidity": 55},
    "izmir": {"city": "Izmir", "temp": 19, "condition": "Clear", "humidity": 48},
    # 일본
    "tokyo": {"city": "Tokyo", "temp": 15, "condition": "Partly Cloudy", "humidity": 55},
    "osaka": {"city": "Osaka", "temp": 17, "condition": "Clear", "humidity": 50},
    "kyoto": {"city": "Kyoto", "temp": 16, "condition": "Clear", "humidity": 48},
    # 동남아
    "bangkok": {"city": "Bangkok", "temp": 34, "condition": "Hot & Humid", "humidity": 80},
    "singapore": {"city": "Singapore", "temp": 31, "condition": "Thunderstorms", "humidity": 85},
    "hanoi": {"city": "Hanoi", "temp": 28, "condition": "Humid", "humidity": 75},
    "bali": {"city": "Bali", "temp": 30, "condition": "Partly Cloudy", "humidity": 78},
    # 유럽
    "paris": {"city": "Paris", "temp": 13, "condition": "Overcast", "humidity": 65},
    "london": {"city": "London", "temp": 10, "condition": "Rainy", "humidity": 80},
    "rome": {"city": "Rome", "temp": 18, "condition": "Sunny", "humidity": 45},
    "barcelona": {"city": "Barcelona", "temp": 17, "condition": "Clear", "humidity": 55},
    "berlin": {"city": "Berlin", "temp": 8, "condition": "Cloudy", "humidity": 60},
    "amsterdam": {"city": "Amsterdam", "temp": 9, "condition": "Rainy", "humidity": 75},
    "prague": {"city": "Prague", "temp": 7, "condition": "Overcast", "humidity": 62},
    "vienna": {"city": "Vienna", "temp": 10, "condition": "Partly Cloudy", "humidity": 55},
    "zurich": {"city": "Zurich", "temp": 8, "condition": "Clear", "humidity": 50},
    # 미주
    "new york": {"city": "New York", "temp": 11, "condition": "Clear", "humidity": 50},
    "los angeles": {"city": "Los Angeles", "temp": 22, "condition": "Sunny", "humidity": 30},
    "miami": {"city": "Miami", "temp": 28, "condition": "Humid", "humidity": 75},
    "vancouver": {"city": "Vancouver", "temp": 9, "condition": "Rainy", "humidity": 78},
    "cancun": {"city": "Cancún", "temp": 30, "condition": "Sunny", "humidity": 70},
    # 오세아니아
    "sydney": {"city": "Sydney", "temp": 20, "condition": "Clear", "humidity": 55},
    "melbourne": {"city": "Melbourne", "temp": 16, "condition": "Changeable", "humidity": 60},
    "auckland": {"city": "Auckland", "temp": 17, "condition": "Partly Cloudy", "humidity": 65},
    # 중동/아프리카
    "dubai": {"city": "Dubai", "temp": 35, "condition": "Hot & Sunny", "humidity": 40},
    "cairo": {"city": "Cairo", "temp": 28, "condition": "Sunny", "humidity": 25},
    "cape town": {"city": "Cape Town", "temp": 22, "condition": "Clear", "humidity": 50},
    "marrakech": {"city": "Marrakech", "temp": 25, "condition": "Sunny", "humidity": 30},
}


@tool
def get_weather(city: str) -> str:
    """Get current weather for any major city worldwide.
    Examples: Seoul, Istanbul, Tokyo, Paris, New York, Bangkok, Dubai"""
    try:
        city_key = city.strip().lower()

        weather = WEATHER_DATA.get(city_key) or WEATHER_DATA.get(city.strip())

        if not weather:
            return (
                f"'{city}' weather data not available in our database.\n"
                f"Try major cities like: Seoul, Tokyo, Paris, Istanbul, New York, Bangkok, Dubai, London, Sydney"
            )

        return (
            f"🌤 {weather['city']} weather:\n"
            f"  Temperature: {weather['temp']}°C\n"
            f"  Condition: {weather['condition']}\n"
            f"  Humidity: {weather['humidity']}%"
        )

    except Exception as e:
        return f"Weather lookup error: {str(e)}"
