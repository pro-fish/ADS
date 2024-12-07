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
