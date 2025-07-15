from . import map_toolkit # Ensure this import is correct based on your project structure
# This file contains the API toolkit for verifying various aspects of campus maps.
import json
from collections import defaultdict
import re
from pathlib import Path
class VerificationToolkit:
    # 1. **【关键修改】** 构造函数现在接收一个 `topology_data_path`
    def __init__(self, topology_data_path: Path):
        """
        Initializes the VerificationToolkit.
        
        Args:
            topology_data_path (Path): The absolute path to the 'topology_sample' directory.
        """
        print("Initializing Verification Toolkit: Loading map data...")
        
        # 2. **【关键修改】** 使用传入的路径来构建具体的文件路径
        self.topology_path = topology_data_path
        campus_map_path = self.topology_path / 'campus.txt'
        self.connectivity_path = self.topology_path / 'connectivity.txt'

        if not campus_map_path.is_file() or not self.connectivity_path.is_file():
            raise FileNotFoundError(f"Core map files not found in '{self.topology_path}'.")

        # 3. 加载逻辑保持不变，但使用的是绝对路径
        self.outdoor_collection = map_toolkit.get_outdoor_nodes(str(campus_map_path))
        print("Map data loaded successfully.")


    def road_paths_verify(self, start_location: str, end_location: str) -> str:
        # 4. **【关键修改】** 调用底层函数时，传递正确的绝对路径
        try:
            optional_paths_names = map_toolkit.optional_road_path_search(
                self.outdoor_collection,
                start_location,
                end_location,
                osm_file=str(self.connectivity_path) # 把正确的路径传进去
            )
            if not optional_paths_names:
                return json.dumps({
                    "start_location": start_location,
                    "end_location": end_location,
                    "paths_found": 0,
                    "paths": []
                }, indent=2)
                
            # Step 2: Process each path to get nodes and calculate length.
            result_paths = []
            for path_names in optional_paths_names:
                # Get the physical nodes for the path using your provided logic
                path_nodes = map_toolkit.road_path_nodes_search(self.outdoor_collection, path_names)

                # Calculate the physical length
                path_length = map_toolkit.road_path_length_calculation(path_nodes)

                # Format the result for this path
                result_paths.append({
                    "road_segments": path_names,
                    "length": round(path_length, 2),
                    "start_door": path_names[0],
                    "end_door": path_names[-1]
                })
                
            # Step 3: Sort paths by length and add a final path_id.
            result_paths.sort(key=lambda x: x['length'])
            for i, p in enumerate(result_paths):
                p['path_id'] = f"path_{i+1}"
                
            return json.dumps({
                "start_location": start_location,
                "end_location": end_location,
                "paths_found": len(result_paths),
                "paths": result_paths
            }, indent=2)

        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, indent=2)



    def cameras_verify(self, location_name: str, scenario_name: str = None) -> str:
        # This function is correct and remains unchanged.
        try:
            if scenario_name:
                file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
                print(f"Loading indoor nodes from: {file_path}")
                collection = map_toolkit.get_indoor_nodes(file_path)
                # print(collection.locations)
                scenario_name = scenario_name.replace('_', '-').lower()
                location_obj = collection.get_location(scenario_name)
                name = scenario_name
            else:
                collection = map_toolkit.get_outdoor_nodes('./campus.txt')
                location_obj = collection.get_location(location_name)
                name = location_name
            if not location_obj: return json.dumps({"error": f"Location or scenario '{name}' not found."})
            return json.dumps({"location": name, "camera_count": len(location_obj.cameras)})
        except Exception as e: return json.dumps({"error": str(e)})

    def doors_verify(self, location_name: str, scenario_name: str = None) -> str:
        # This function is correct and remains unchanged.
        try:
            if scenario_name:
                scenario_name = scenario_name.replace('_', '-').lower()
                file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
                collection = map_toolkit.get_indoor_nodes(file_path)
                location_obj = collection.get_location(scenario_name)
                name = scenario_name
            else:
                collection = map_toolkit.get_outdoor_nodes('./campus.txt')
                location_obj = collection.get_location(location_name)
                name = location_name
            if not location_obj: return json.dumps({"error": f"Location or scenario '{name}' not found."})
            door_names = [d.tags.get('door', 'unnamed_door') for d in location_obj.doors]
            return json.dumps({"location": name, "doors": door_names})
        except Exception as e: return json.dumps({"error": str(e)})

    def facilities_verify(self, location_name: str, scenario_name: str) -> str:
        # This function is correct and remains unchanged.
        try:
            file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
            collection = map_toolkit.get_indoor_nodes(file_path)
            scenario_name = scenario_name.replace('_', '-').lower()
            location_obj = collection.get_location(scenario_name)
            if not location_obj: return json.dumps({"error": f"Scenario '{scenario_name}' not found."})
            facility_counts = defaultdict(int)
            for node in location_obj.facilities:
                base_name = re.sub(r'_\d+$', '', node.tags.get('facility', 'unnamed'))
                facility_counts[base_name] += 1
            return json.dumps({"scenario": scenario_name, "facilities": dict(facility_counts)})
        except Exception as e: return json.dumps({"error": str(e)})

    def elevators_verify(self, location_name: str) -> str:
        # This function is correct and remains unchanged.
        try:
            file_path = self.topology_path / f"{location_name.replace(' ', '_').lower()}.txt"
            collection = map_toolkit.get_indoor_nodes(file_path)
            hall_loc = next((loc for name, loc in collection.locations.items() if 'hall' in name.lower()), None)
            if not hall_loc: return json.dumps({"location": location_name, "elevators": []})
            elevator_names = [e.tags.get('elevator', 'unnamed') for e in hall_loc.elevators]
            return json.dumps({"location": location_name, "elevators": elevator_names})
        except Exception as e: return json.dumps({"error": str(e)})
