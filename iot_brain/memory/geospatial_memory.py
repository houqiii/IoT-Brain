# iot_brain/memory/geospatial_memory.py

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

class GeospatialMemory:
    """
    Manages a persistent, file-based memory for verified geospatial and topological information.

    This class provides a simple key-value store where keys are derived from the
    core attributes of a location, and values are dictionaries containing all
    verified facts about that location (e.g., camera counts, facility names).
    The memory is automatically loaded from a JSON file on initialization and
    saved to the same file whenever it's updated. This ensures that knowledge
    gained from API verifications is preserved across sessions.
    """

    def __init__(self, memory_file_path: str | Path = "geospatial_memory.json"):
        """
        Initializes the GeospatialMemory instance.

        Args:
            memory_file_path (str | Path): The path to the JSON file where the memory
                                           will be stored. Defaults to "geospatial_memory.json"
                                           in the current working directory.
        """
        self.memory_path = Path(memory_file_path)
        self.memory_data: Dict[str, Any] = self._load()

    def _create_key(self, building: Optional[str], floor: Optional[str], semantic_name: str, type: str) -> str:
        """
        Creates a consistent, unique string key from the core attributes of a location.

        Handles None values to ensure key consistency. The order of attributes is fixed.

        Args:
            building (Optional[str]): The name of the building.
            floor (Optional[str]): The floor level (e.g., '1F').
            semantic_name (str): The specific name of the location (e.g., 'public classroom 1F 1').
            type (str): The type of the location ('indoor' or 'outdoor').

        Returns:
            str: A unique composite key for the memory dictionary.
        """
        # Standardize None values to a string representation for key consistency
        b = building if building is not None else "null"
        f = floor if floor is not None else "null"
        # The key is a simple concatenation with a clear separator
        return f"{b}|{f}|{semantic_name}|{type}"

    def _load(self) -> Dict[str, Any]:
        """
        Loads the memory data from the JSON file.

        If the file does not exist or is corrupted, it returns an empty dictionary.

        Returns:
            Dict[str, Any]: The loaded memory data.
        """
        if not self.memory_path.exists():
            return {}
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If the file is corrupted or gone, start with a fresh memory
            return {}

    def _save(self):
        """
        Saves the current state of the memory to the JSON file.
        The JSON is saved in a human-readable format.
        """
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_data, f, indent=4, ensure_ascii=False)

    def query_memory(self, building: Optional[str], floor: Optional[str], semantic_name: str, type: str) -> Optional[Dict[str, Any]]:
        """
        Queries the memory for previously verified information about a location.

        This should be called before making an API call to avoid redundant verification.

        Args:
            building (Optional[str]): The building name of the target location.
            floor (Optional[str]): The floor of the target location.
            semantic_name (str): The semantic name of the target location.
            type (str): The type of the target location.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing all verified facts for the
                                      location if found, otherwise None.
        """
        key = self._create_key(building, floor, semantic_name, type)
        return self.memory_data.get(key)

    def add_verified_info(self, building: Optional[str], floor: Optional[str], semantic_name: str, type: str, new_info: Dict[str, Any]):
        """
        Adds or updates verified information for a location and saves it to the file.

        This method is called after a successful API verification. It merges the
        new information with any existing information for that location.

        Args:
            building (Optional[str]): The building name.
            floor (Optional[str]): The floor level.
            semantic_name (str): The semantic name of the location.
            type (str): The location type ('indoor' or 'outdoor').
            new_info (Dict[str, Any]): A dictionary containing the newly verified
                                       fact(s) (e.g., {'camera_count': 2}).
        """
        key = self._create_key(building, floor, semantic_name, type)

        if key in self.memory_data:
            # If entry exists, update it with the new info
            self.memory_data[key].update(new_info)
        else:
            # If entry does not exist, create a new one based on the standard structure
            # and add the new info. The 'id' and 'specific_facilities' can be populated
            # later or kept as placeholders.
            self.memory_data[key] = {
                "id": "node_placeholder", # Placeholder, can be updated if an ID is ever verified
                "semantic_name": semantic_name,
                "floor": floor,
                "building": building,
                "type": type,
                "specific_facilities": [], # Placeholder
                "verified_properties": new_info
            }
            # Merge new_info into the main level for direct access
            self.memory_data[key].update(new_info)

        self._save()
        print(f"[GeospatialMemory] Updated memory for key '{key}' with: {new_info}")


if __name__ == '__main__':
    # --- DEMONSTRATION OF USAGE ---
    
    # Create an instance of the memory. This will create 'geospatial_memory.json' if it doesn't exist.
    memory = GeospatialMemory(memory_file_path="geospatial_memory.json")
    print("--- Initializing Memory ---")
    print(f"Memory loaded from: {memory.memory_path.resolve()}")
    
    # Define a location to check
    building = 'library'
    floor = '1F'
    semantic_name = 'public_classroom_1F_1'
    location_type = 'indoor'

    # 1. Query for information that doesn't exist yet
    print("\n--- Step 1: Querying non-existent info ---")
    cached_data = memory.query_memory(building, floor, semantic_name, location_type)
    print(f"Query Result: {cached_data}")
    if cached_data is None:
        print("Info not in memory. An API call would be needed.")

    # 2. Simulate a successful API call and add the verified info to memory
    print("\n--- Step 2: Adding verified camera count ---")
    verified_camera_info = {'camera_count': 2}
    memory.add_verified_info(building, floor, semantic_name, location_type, verified_camera_info)

    # 3. Query again to see the new data
    print("\n--- Step 3: Querying again ---")
    cached_data = memory.query_memory(building, floor, semantic_name, location_type)
    print(f"Query Result: {json.dumps(cached_data, indent=2)}")
    if cached_data and cached_data.get('camera_count') == 2:
        print("Successfully retrieved camera count from memory. No API call needed for this fact.")

    # 4. Simulate another API call for the same location and add facility info
    print("\n--- Step 4: Adding verified facility info to the same location ---")
    verified_facility_info = {'facilities': ['public_classroom_1F_1_students_desk', 'public_classroom_1F_1_teacher_lectern']}
    memory.add_verified_info(building, floor, semantic_name, location_type, verified_facility_info)

    # 5. Query a final time to see the merged data
    print("\n--- Step 5: Final query showing merged data ---")
    cached_data = memory.query_memory(building, floor, semantic_name, location_type)
    print(f"Final Merged Data: {json.dumps(cached_data, indent=2)}")
    print("\nDemonstration complete. Check the 'geospatial_memory.json' file.")