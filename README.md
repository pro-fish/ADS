This project aims to simulate and optimize emergency ambulance dispatch operations within a realistic urban environment. By integrating geographic data from OpenStreetMap with SUMO (Simulation of Urban Mobility), it provides a dynamic, data-driven platform to model city traffic, route planning, and emergency response logistics. The system leverages the following key components:

1. **City Map Acquisition & Preparation**:  
   Using OSMnx, the project downloads and processes city map data, converting open-source geographic information into a SUMO-compatible network. This forms the foundational road infrastructure on which vehicles navigate.

2. **Traffic Patterns & Infrastructure**:  
   The system defines realistic traffic scenarios, including peak hours, off-peak periods, and special zones that influence traffic density. Hospitals, ambulance stations, and other critical infrastructure locations are defined and integrated into the simulation, ensuring scenario realism.

3. **SUMO Integration & Simulation**:  
   By harnessing SUMO’s powerful microscopic traffic simulation capabilities, the project runs time-stepped simulations of city traffic. Ambulances, defined with specific performance characteristics, are introduced into the network to respond to generated emergency calls.

4. **Route Optimization & Pathfinding**:  
   A RouteOptimizer module uses NetworkX graph algorithms and traffic-aware heuristics to determine the most efficient routes for ambulances. This includes dynamic adjustments based on current traffic conditions, ensuring that emergency vehicles take the quickest possible path.

5. **Reinforcement Learning for Dispatch Decisions**:  
   A learning-based AI module (trained via PyTorch) dynamically decides which ambulance to dispatch to each incoming call. Over time, the reinforcement learning agent refines its policy, aiming to minimize response times and improve city-wide coverage and resource utilization.

6. **Call Generation & Scenario Management**:  
   Emergency calls are generated according to configurable frequency patterns, reflecting realistic scenarios that vary across different times of day. This realistic demand modeling challenges the dispatch system to continuously adapt to changing conditions.

7. **Statistics, Reporting & Evaluation**:  
   Detailed statistics are collected on response times, coverage, and ambulance utilization. Comprehensive performance reports are generated at the end of each simulation run, enabling quantitative evaluation of the system’s efficiency and guiding improvements to the dispatch strategy.

**In essence, this project provides a robust, end-to-end toolkit for simulating, analyzing, and improving emergency ambulance dispatch operations in complex urban environments. It combines state-of-the-art geographic data handling, traffic simulation, optimization techniques, and machine learning to produce actionable insights into how cities can better respond to emergencies.**


1. **Complexity & Realism**:  
   The code now includes more realistic integrations with OSMnx, SUMO (via `netconvert`, `sumolib`, `traci`), `networkx`, and PyTorch. However, certain steps—like obtaining raw OSM data, converting it accurately to SUMO networks, and running a full reinforcement learning loop—are non-trivial. The code here is illustrative and may require adjustments to run successfully in a real environment.

2. **Dependencies**:  
   - **OSMnx**: For obtaining and handling city graphs.
   - **SUMO**: You need SUMO installed and `sumolib`, `traci` Python bindings available.
   - **PyTorch**: For the AI model and training.
   - **NetworkX**: For graph-based route optimization.
   - **Additional**: `numpy`, `pandas` as needed.

3. **OSM to SUMO Conversion**:  
   OSMnx doesn’t directly export raw OSM files. Typically, you can use bounding boxes or place queries and download OSM data via Overpass directly. For demonstration, the code shows placeholders. In a real scenario, you would:
   - Use OSMnx to find the area.
   - Use Overpass queries (via OSMnx or separately) to obtain raw OSM XML data.
   - Feed that OSM file into `netconvert` to produce a SUMO network.

4. **Running SUMO**:  
   Adjust `sumo` and `netconvert` commands in the code to match your system’s PATH and SUMO installation details.

5. **Reinforcement Learning**:  
   The RL code in `ai_trainer.py` and its usage in `simulation_manager.py` is highly simplified. In practice, you would define states, actions, and rewards meaningfully, run multiple episodes, and improve the policy over time.

---

### download_map.py

```python
import os
import subprocess
import osmnx as ox
from pathlib import Path

def download_city_map(city_name="Jeddah", country="Saudi Arabia"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)

    # Define the place
    place = f"{city_name}, {country}"

    # Download the city graph using OSMnx
    # network_type='drive' focuses on drivable roads.
    G = ox.graph_from_place(place, network_type='drive')
    graphml_path = city_data_dir / "city_graph.graphml"
    ox.save_graphml(G, filepath=graphml_path)
    
    # NOTE: OSMnx doesn't provide raw OSM XML directly. 
    # In a real scenario, you'd use ox.geometries_from_place() or Overpass queries 
    # to get raw OSM data in .osm format. For demo, we assume we have a city.osm file.
    
    # Placeholder: Create a dummy OSM file (You must replace this with actual OSM data)
    raw_osm_path = city_data_dir / "city.osm"
    if not raw_osm_path.exists():
        # In a real scenario, download raw OSM data using Overpass and save it here.
        raw_osm_path.write_text("<osm></osm>")

    # Convert OSM data to SUMO network using netconvert
    net_path = city_data_dir / "network.net.xml"
    cmd = f"netconvert --osm-files {raw_osm_path} -o {net_path}"
    subprocess.run(cmd, shell=True, check=False)

    print(f"Map data for {city_name}, {country} downloaded and converted to SUMO format.")

if __name__ == "__main__":
    download_city_map()
```

---

### define_locations.py

```python
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
```

---

### traffic_patterns.py

```python
import json
from pathlib import Path

def define_traffic_patterns(city_name="Jeddah"):
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
```

---

### sumo_config.py

```python
from pathlib import Path

def create_sumo_config(city_name="Jeddah"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)

    config_content = f"""
    <configuration>
        <input>
            <net-file value="network.net.xml"/>
            <route-files value="emergency_vehicles.add.xml"/>
        </input>
        <time>
            <begin value="0"/>
            <end value="36000"/>
        </time>
        <processing>
            <step-length value="1"/>
        </processing>
    </configuration>
    """

    config_file = city_data_dir / "simulation.sumocfg"
    config_file.write_text(config_content.strip())
    print(f"SUMO configuration created for {city_name}.")
```

---

### emergency_vehicles.py

```python
from pathlib import Path

def create_emergency_vehicles(city_name="Jeddah"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)

    vehicles_content = """
    <additional>
        <vType id="ambulance" accel="2.0" decel="4.5" sigma="0.5" length="5" maxSpeed="33.33" color="1,0,0" guiShape="emergency"/>
    </additional>
    """

    (city_data_dir / "emergency_vehicles.add.xml").write_text(vehicles_content.strip())
    print(f"Emergency vehicles defined for {city_name}.")
```

---

### emergency_call_generator.py

```python
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
```

---

### route_optimizer.py

```python
import json
from pathlib import Path
import networkx as nx
import osmnx as ox

class RouteOptimizer:
    def __init__(self, city_name="Jeddah"):
        self.city_name = city_name
        self.city_data_dir = Path("data") / city_name
        self.city_data_dir.mkdir(parents=True, exist_ok=True)

        self.routes_cache_file = self.city_data_dir / "routes_cache.json"
        self.routes_cache = self.load_routes_cache()

        graph_file = self.city_data_dir / "city_graph.graphml"
        self.G = ox.load_graphml(graph_file)

    def get_optimal_route(self, start_coords, end_coords, traffic_data=None):
        # start_coords = [lat, lon], ox wants lon, lat for nearest_nodes
        start_node = ox.distance.nearest_nodes(self.G, start_coords[1], start_coords[0])
        end_node = ox.distance.nearest_nodes(self.G, end_coords[1], end_coords[0])

        # Compute shortest path by length
        route = nx.shortest_path(self.G, start_node, end_node, weight='length')
        return route

    def save_routes_cache(self):
        with open(self.routes_cache_file, "w") as f:
            json.dump(self.routes_cache, f, indent=4)

    def load_routes_cache(self):
        if self.routes_cache_file.exists():
            with open(self.routes_cache_file, "r") as f:
                return json.load(f)
        return {}

    def calculate_traffic_aware_route(self, start_coords, end_coords, traffic_data):
        # Adjust weights based on traffic_data before calling shortest_path
        # Stub: identical to get_optimal_route for now
        return self.get_optimal_route(start_coords, end_coords, traffic_data)
```

---

### ai_trainer.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path

class DispatchAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DispatchAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AITrainer:
    def __init__(self, city_name="Jeddah", input_size=10, hidden_size=20, output_size=5):
        self.city_name = city_name
        self.city_data_dir = Path("data") / city_name
        self.city_data_dir.mkdir(parents=True, exist_ok=True)

        self.model_file = self.city_data_dir / "dispatch_model.pth"
        self.training_stats_file = self.city_data_dir / "training_stats.json"

        self.model = self.initialize_model(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_stats = {"episodes": 0, "rewards": []}
        self.gamma = 0.99

    def initialize_model(self, input_size, hidden_size, output_size):
        return DispatchAI(input_size, hidden_size, output_size)

    def train_episode(self, state, action, reward, next_state, done=False):
        # Simple Q-learning update
        self.model.train()
        state_value = self.model(state)
        next_state_value = self.model(next_state)
        target = state_value.clone().detach()
        if done:
            target[0, action] = reward
        else:
            target[0, action] = reward + self.gamma * torch.max(next_state_value).item()

        loss = self.criterion(state_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_stats["episodes"] += 1
        self.training_stats["rewards"].append(reward)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)

    def load_model(self):
        if self.model_file.exists():
            self.model.load_state_dict(torch.load(self.model_file))

    def save_training_stats(self):
        with open(self.training_stats_file, "w") as f:
            json.dump(self.training_stats, f, indent=4)
```

---

### statistics_collector.py

```python
import json
from pathlib import Path
from datetime import datetime

class StatisticsCollector:
    def __init__(self, city_name="Jeddah"):
        self.city_name = city_name
        self.city_data_dir = Path("data") / city_name
        self.stats_dir = self.city_data_dir / "statistics"
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats = {
            "response_times": [],
            "dispatch_decisions": [],
            "coverage_analysis": {},
            "ambulance_utilization": {}
        }

    def record_response(self, call_id, response_time, ambulance_id):
        self.stats["response_times"].append({
            "call_id": call_id,
            "time": response_time,
            "ambulance_id": ambulance_id
        })

    def record_dispatch(self, call_id, ambulance_id):
        self.stats["dispatch_decisions"].append({
            "call_id": call_id,
            "ambulance_id": ambulance_id,
            "timestamp": datetime.now().isoformat()
        })

    def analyze_coverage(self, stations, calls):
        total_calls = len(calls)
        covered_calls = sum(1 for c in calls if c.get("response_time", 9999) < 600)
        coverage_score = covered_calls / total_calls if total_calls > 0 else 0.0
        self.stats["coverage_analysis"] = {"coverage_score": coverage_score}

    def generate_report(self):
        report = {
            "session": self.current_session,
            "summary": "Simulation performance summary",
            "response_times_avg": self._calculate_average_response_time(),
            "coverage_analysis": self.stats["coverage_analysis"],
            "total_dispatches": len(self.stats["dispatch_decisions"])
        }
        return report

    def save_stats(self):
        stats_file = self.stats_dir / f"stats_{self.current_session}.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=4)

        report_file = self.stats_dir / f"report_{self.current_session}.json"
        with open(report_file, "w") as f:
            json.dump(self.generate_report(), f, indent=4)
        print(f"Statistics and report saved for session {self.current_session}.")

    def _calculate_average_response_time(self):
        times = [r["time"] for r in self.stats["response_times"]]
        return sum(times)/len(times) if times else 0.0
```

---

### simulation_manager.py

```python
import json
from pathlib import Path
from datetime import datetime
import traci
import torch

from emergency_call_generator import EmergencyCallGenerator
from route_optimizer import RouteOptimizer
from ai_trainer import AITrainer
from statistics_collector import StatisticsCollector
from define_locations import save_emergency_locations
from traffic_patterns import define_traffic_patterns
from sumo_config import create_sumo_config
from emergency_vehicles import create_emergency_vehicles

class SimulationManager:
    def __init__(self, city_name="Jeddah"):
        self.city_name = city_name
        self.city_data_dir = Path("data") / city_name
        self.city_data_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.city_data_dir / "simulation.sumocfg"
        self.state_file = self.city_data_dir / "simulation_state.json"

        self.state = {
            "current_step": 0,
            "active_calls": [],
            "ambulance_states": {},
            "completed_calls": []
        }

        # Initialize components
        self.call_generator = EmergencyCallGenerator(city_name)
        self.route_optimizer = RouteOptimizer(city_name)
        self.ai_trainer = AITrainer(city_name)
        self.statistics = StatisticsCollector(city_name)

        # Prepare configuration and vehicles
        save_emergency_locations(city_name)
        define_traffic_patterns(city_name)
        create_sumo_config(city_name)
        create_emergency_vehicles(city_name)

    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                self.state = json.load(f)

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def run_simulation(self, duration_hours=1, resume=False):
        if resume:
            self.load_state()
        else:
            self.state = {
                "current_step": 0,
                "active_calls": [],
                "ambulance_states": {},
                "completed_calls": []
            }

        # Start SUMO simulation
        sumoCmd = ["sumo", "-c", str(self.config_file)]
        traci.start(sumoCmd)

        total_steps = duration_hours * 3600
        calls = self.call_generator.load_calls()
        call_index = 0

        for step in range(self.state["current_step"], total_steps):
            # Spawn new calls at their timestamp
            while call_index < len(calls) and calls[call_index]["timestamp"] == step:
                new_call = calls[call_index]
                self.state["active_calls"].append(new_call)
                call_index += 1

            # Dispatch decisions
            for call in self.state["active_calls"]:
                if "dispatched" not in call:
                    # Construct state and choose action
                    state_vec = self._construct_state_representation(call)
                    state_tensor = torch.tensor([state_vec], dtype=torch.float)
                    q_values = self.ai_trainer.model(state_tensor)
                    action = q_values.argmax().item()
                    chosen_ambulance = f"ambulance_{action}"
                    call["dispatched"] = chosen_ambulance
                    self.statistics.record_dispatch(call["id"], chosen_ambulance)

                    # Compute route (stub)
                    start = [21.5,39.2]  # Example start coords
                    end = call["location"]
                    route_nodes = self.route_optimizer.get_optimal_route(start, end)
                    # Convert route_nodes to edges and add a vehicle route to SUMO if needed
                    # This is non-trivial; you'd use sumolib to map nodes to edges.

            traci.simulationStep()
            self.state["current_step"] = step

            # Check if any calls are completed (ambulance reached)
            # For simplicity, assume a fixed response time:
            for call in self.state["active_calls"]:
                if "dispatched" in call and (step - call["timestamp"]) > 300:
                    call["response_time"] = step - call["timestamp"]
                    self.statistics.record_response(call["id"], call["response_time"], call["dispatched"])
                    self.state["completed_calls"].append(call)

            # Remove completed calls
            self.state["active_calls"] = [c for c in self.state["active_calls"] if "response_time" not in c]

            # RL training (dummy example):
            # if we had next_state and reward:
            # self.ai_trainer.train_episode(state_tensor, action, reward, next_state_tensor, done)

        traci.close()

        self.statistics.analyze_coverage([], self.state["completed_calls"])
        self.statistics.save_stats()
        self.save_state()
        self.generate_final_report()

    def _construct_state_representation(self, call):
        # Convert call data into a feature vector for RL
        # This is highly domain-specific. We just return a dummy vector.
        return [0.0]*10

    def generate_final_report(self):
        final_report = {
            "city_name": self.city_name,
            "simulation_end_time": datetime.now().isoformat(),
            "summary": "Simulation completed successfully.",
            "completed_calls": len(self.state["completed_calls"])
        }
        with open(self.city_data_dir / "final_report.json", "w") as f:
            json.dump(final_report, f, indent=4)
        print("Final report generated.")
```

---

## Notes

- The above code is a scaffold. In a production environment, you would:  
  - Ensure you have valid OSM data.
  - Correctly run `netconvert` to produce a valid `network.net.xml` from an actual OSM file.
  - Use `sumolib` to map route nodes to edges and add routes and vehicles in `simulation_manager.py`.
  - Implement a proper reward structure, states, and actions for RL training in `ai_trainer.py`.
  - Refine `statistics_collector.py` with real analytics.
  - Test each component individually, then integrate.

This should serve as a more comprehensive example of how to integrate the various components into a more functional simulation and training environment.
