import json
import random
from pathlib import Path

class EmergencyCallGenerator:
    def __init__(self, city_name="Jeddah"):
        self.city_name = city_name
        self.city_data_dir = Path("data") / city_name
        self.city_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.bounds = {
            "min_lat": 21.4,
            "max_lat": 21.6,
            "min_lon": 39.1,
            "max_lon": 39.3
        }
        
        self.call_counter = 0
        self.calls_file = self.city_data_dir / "generated_calls.json"

    def generate_calls(self, duration_hours=1, frequency=5):
        calls = []
        for hour in range(duration_hours):
            current_frequency = self.adjust_frequency_for_hour(hour, frequency)
            for _ in range(current_frequency):
                timestamp = hour * 3600 + random.randint(0, 3599)
                call = self.generate_single_call(timestamp)
                calls.append(call)
        self.save_calls(calls)

    def generate_single_call(self, timestamp):
        lat = random.uniform(self.bounds["min_lat"], self.bounds["max_lat"])
        lon = random.uniform(self.bounds["min_lon"], self.bounds["max_lon"])
        self.call_counter += 1
        return {
            "id": f"call_{self.call_counter}",
            "timestamp": timestamp,
            "location": [lat, lon],
            "severity": random.choice(["low", "medium", "high"])
        }

    def save_calls(self, calls):
        with open(self.calls_file, "w") as f:
            json.dump(calls, f, indent=4)
        print(f"Generated calls saved for {self.city_name}.")

    def load_calls(self):
        if self.calls_file.exists():
            with open(self.calls_file, "r") as f:
                return json.load(f)
        return []

    def adjust_frequency_for_hour(self, hour, base_frequency):
        if 7 <= hour < 9 or 16 <= hour < 19:
            return int(base_frequency * 1.5)
        return base_frequency
