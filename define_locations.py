import json
from pathlib import Path

def save_emergency_locations(city_name="Jeddah"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)
    
    locations = {
        "ambulance_stations": [
            {"id": "as1", "name": "Central Station", "coords": [21.5, 39.2]},
            {"id": "as2", "name": "East Station", "coords": [21.51, 39.3]}
        ],
        "hospitals": [
            {"id": "h1", "name": "General Hospital", "coords": [21.52, 39.25]},
            {"id": "h2", "name": "City Clinic", "coords": [21.49, 39.23]}
        ]
    }

    with open(city_data_dir / "emergency_locations.json", "w") as f:
        json.dump(locations, f, indent=4)
    print(f"Emergency locations saved for {city_name}.")
if __name__ == "__main__":
    save_emergency_locations()
