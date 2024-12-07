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
    def __init__(self, city_name="Makkah Region"):
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
if __name__ == "__main__":
    manager = SimulationManager()
    # Example: Run a short simulation
    manager.run_simulation(duration_hours=1, resume=False)
