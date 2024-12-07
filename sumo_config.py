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
