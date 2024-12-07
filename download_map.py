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

    # STEP 1: Download or use existing city graph
    if graphml_path.exists():
        print(f"Using existing GraphML file at: {graphml_path}")
        G = ox.load_graphml(graphml_path)
    else:
        print("Downloading city graph using OSMnx...")
        G = ox.graph_from_place(place, network_type='drive')
        ox.save_graphml(G, filepath=graphml_path)
        print(f"GraphML file saved at: {graphml_path}")

    # STEP 2: Obtain Raw OSM Data from Overpass API if not available locally
    raw_osm_path = city_data_dir / "city.osm"
    if not raw_osm_path.exists():
        # To fetch raw OSM data, we need a bounding polygon of the place.
        # OSMnx geocode_to_gdf gives a polygon for the place
        gdf = ox.geocode_to_gdf(place)
        if gdf.empty:
            # If we fail to get a polygon, fallback to empty OSM
            print("Failed to retrieve place polygon. Using dummy OSM.")
            raw_osm_path.write_text("<osm></osm>")
        else:
            # Extract boundary polygon and convert to Overpass query area
            minx, miny, maxx, maxy = gdf.total_bounds

            # Construct an Overpass query to fetch highways in the bounding box
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

    # STEP 3: Convert OSM data to SUMO network using netconvert
    net_path = city_data_dir / "network.net.xml"

    # Only run netconvert if the network file doesn't exist
    if not net_path.exists():
        cmd = f"netconvert --osm-files {raw_osm_path} -o {net_path}"
        print("Converting OSM data to SUMO network...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"SUMO network created at: {net_path}")
        else:
            print("netconvert failed to create a SUMO network.")
            print("Error output:", result.stderr)
    else:
        print(f"Using existing SUMO network at: {net_path}")

    print(f"Map data for {city_name}, {country} is prepared.")

if __name__ == "__main__":
    download_city_map()
