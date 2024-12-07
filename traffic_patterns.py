import json
from pathlib import Path

def define_traffic_patterns(city_name="Makkah Region"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)
    
    traffic_patterns = {
        "peak_hours": {
            "morning": {"start": 7, "end": 9, "density": 1.5},
            "evening": {"start": 16, "end": 19, "density": 1.7}
        },
        "off_peak": {"density": 1.0},
        "night": {"start": 22, "end": 5, "density": 0.5},
        "special_zones": [
            {"name": "Downtown", "coords": [21.50, 39.20], "radius": 500, "multiplier": 2.0}
        ]
    }
    
    with open(city_data_dir / "traffic_patterns.json", "w") as f:
        json.dump(traffic_patterns, f, indent=4)
    print(f"Traffic patterns defined for {city_name}.")

if __name__ == "__main__":
    define_traffic_patterns()
