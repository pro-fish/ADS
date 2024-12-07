import os
import subprocess
from pathlib import Path
import requests
import osmnx as ox

def download_city_map(city_name="Makkah Region", country="Saudi Arabia"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)

    place = f"{city_name}, {country}"

    graphml_path = city_data_dir / "city_graph.graphml"

    if graphml_path.exists():
        print(f"Using existing GraphML file at: {graphml_path}")
        G = ox.load_graphml(graphml_path)
    else:
        print("Downloading city graph using OSMnx...")
        G = ox.graph_from_place(place, network_type='drive')
        ox.save_graphml(G, filepath=graphml_path)
        print(f"GraphML file saved at: {graphml_path}")

    raw_osm_path = city_data_dir / "city.osm"
    if not raw_osm_path.exists():
        gdf = ox.geocode_to_gdf(place)
        if gdf.empty:
            print("Failed to retrieve place polygon. Using dummy OSM.")
            raw_osm_path.write_text("<osm></osm>")
        else:
            minx, miny, maxx, maxy = gdf.total_bounds
            overpass_query = f"""
            [out:xml][timeout:180];
            (
              way["highway"]({miny},{minx},{maxy},{maxx});
              relation["highway"]({miny},{minx},{maxy},{maxx});
            );
            (._;>;);
            out body;
            """
            print("Requesting raw OSM data from Overpass...")
            r = requests.get("https://overpass-api.de/api/interpreter", params={'data': overpass_query.strip()})
            if r.status_code == 200 and "<osm" in r.text:
                raw_osm_path.write_bytes(r.content)
                print(f"Raw OSM data saved at: {raw_osm_path}")
            else:
                print("Failed to retrieve OSM data from Overpass. Using dummy OSM file.")
                raw_osm_path.write_text("<osm></osm>")
    else:
        print(f"Using existing OSM file at: {raw_osm_path}")

    net_path = city_data_dir / "network.net.xml"
    if not net_path.exists():
        print("Converting OSM data to SUMO network...")
        # Use a list of arguments rather than a single string
        # Also ensure the paths are properly quoted to handle spaces
        cmd = [
            "netconvert",
            "--osm-files", str(raw_osm_path),
            "-o", str(net_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"SUMO network created at: {net_path}")
        else:
            print("netconvert failed to create a SUMO network.")
            print("Error output:", result.stderr)
    else:
        print(f"Using existing SUMO network at: {net_path}")

    print(f"Map data for {city_name}, {country} downloaded and converted to SUMO format.")

if __name__ == "__main__":
    download_city_map()
