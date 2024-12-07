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
