from pathlib import Path

def create_emergency_vehicles(city_name="Makkah Region"):
    city_data_dir = Path("data") / city_name
    city_data_dir.mkdir(parents=True, exist_ok=True)

    vehicles_content = """
    <additional>
        <vType id="ambulance" accel="2.0" decel="4.5" sigma="0.5" length="5" maxSpeed="33.33" color="1,0,0" guiShape="emergency"/>
    </additional>
    """

    (city_data_dir / "emergency_vehicles.add.xml").write_text(vehicles_content.strip())
    print(f"Emergency vehicles defined for {city_name}.")

if __name__ == "__main__":
    create_emergency_vehicles()
