import json
from pathlib import Path
from datetime import datetime

class StatisticsCollector:
    def __init__(self, city_name="Makkah Region"):
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

if __name__ == "__main__":
    collector = StatisticsCollector()
    # Example usage:
    collector.record_response("call_1", 300, "ambulance_1")
    collector.analyze_coverage([], [{"id": "call_1", "response_time": 300}])
    collector.save_stats()
