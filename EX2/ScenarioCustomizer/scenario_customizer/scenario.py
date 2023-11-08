import json
from importlib.resources import files
from typing import Optional
from random import randint


topogrophy_objects_with_id = [
        "obstacles", "measurementAreas", "stairs", "targets", "targetChangers",
        "absorbingAreas", "aerosolClouds", "droplets", "sources", "dynamicElements"
    ]

required_property_tree = {
    "name": {},
    "scenario": {
        "topography": {
            object_list_name: list for object_list_name in topogrophy_objects_with_id
        }
    }
}

def assert_valid_scenario_spec(prop, scenario_spec, scenario) -> Optional[tuple[str, str]]:
    if type(scenario_spec) is type:
        if type(scenario) is scenario_spec:
            return None

        return (f"The property '{prop}' does not contained the expected type '{scenario_spec}'. "
                "Expected path: "), prop

    if scenario_spec == scenario:
        return None

    return (f"The property '{prop}' does not contained the expected value:\n"
            f"{scenario_spec}\n"
             "Expected path: "), prop



def assert_valid_scenario_recursive(properties_tree: dict, scenario_tree: dict) -> Optional[tuple[str, str]]:
    for prop in properties_tree:
        if prop not in scenario_tree:
            return f"The property '{prop}' was not found. Expected path: ", prop

        if type(properties_tree[prop]) is dict:
            result = assert_valid_scenario_recursive(properties_tree[prop], scenario_tree[prop])
        elif properties_tree[prop] is None:
            result = None
        else:
            result = assert_valid_scenario_spec(prop, properties_tree[prop], scenario_tree[prop])


        if result is not None:
            return result[0], f"{prop}.{result[1]}"

    return None


class InvalidScenarioError(Exception):

    def __init__(
        self,
        error_cause_message: str,
        scenario_name: str =None,
        scenario_path: str =None
        ) -> None:
        error_message = "Invalid scenario"

        if scenario_name is not None:
            error_message += f", named {scenario_name},"
        elif scenario_path is not None:
            error_message += f", in {scenario_path},"

        self.error_message = error_message + f" due to the following error:\n{error_cause_message}"

    def __str__(self) -> str:
        return self.error_message



class Scenario:
    """
    A class that holds the json data of a scenario file. 
    Provides a simple interface to manipulate and validate 
    the scenario file.
    """

    def __init__(self, path: str) -> None:
        """
        Generates a new scenario instance containing 
        the specified scenario file's inforamtion.

        Args:
            path (str): The path to the scenario file.
        """
        self.scenario_path = path
        with open(path, 'r') as file:
            self.scenario_json = json.load(file)

        self.assert_valid_scenario()
        # TODO check the name of all scenarios to create a unique one
        self.scenario_json["name"] = self.scenario_json["name"] + f"_{randint(0,100)}"

    @property
    def topography(self) -> dict:
        """The topography dictionary for the current scenario.

        Returns:
            dict: The topography dictionary for the current scenario.
        """
        return self.scenario_json["scenario"]["topography"]

    @property
    def new_id(self) -> int:
        """The the current id for a new element.

        Returns:
            int: The current id for a new element.
        """
        max_id = 0
        for obj_list in topogrophy_objects_with_id:
            for topogrophy_object in self.topography[obj_list]:
                if "id" in topogrophy_object and topogrophy_object["id"] > max_id:
                    max_id = topogrophy_object["id"]

                elif "attributes" in topogrophy_object \
                 and "id" in topogrophy_object["attributes"] \
                 and topogrophy_object["attributes"]["id"] > max_id:
                    max_id = topogrophy_object["attributes"]["id"]

        return max_id +1

    def assert_valid_scenario(self) -> None:
        """Checks if the current scenario file is valid.

        Raises:
            InvalidScenarioError: If the scenario file provided is invalid.
        """
        result = assert_valid_scenario_recursive(required_property_tree, self.scenario_json)

        if result is not None:
            error_cause_message = result[0] + result[1]
            if "name" in self.scenario_json:
                raise InvalidScenarioError(
                    error_cause_message,
                    scenario_name=self.scenario_json["name"]
                )
            else:
                raise InvalidScenarioError(
                    error_cause_message, 
                    scenario_path=self.scenario_path
                )

    def save(self, path: str):
        """Saves the current version of the scenario in the provided path.

        Args:
            path (str): The path in which to save the scenario.

        Raises:
            FileExistsError: If the path is the one for the scenario being edited.
        """
        if path == self.scenario_path:
            raise FileExistsError()

        with open(path, 'w') as file:
            json.dump(self.scenario_json, file, indent=2, separators=(',', ' : '))

    def add_pedestrian(self, x: float, y: float, target_ids: set[int]) -> Optional[int]:
        """Adds a new pedestrian without source to the scenario at position (x,y).

        Args:
            x (float): The position of the new pedestrian on the x axis.
            y (float): The position of the new pedestrian on the y axis.
            target_ids (list[int]): The ids of this pedestrian's targets.
        
        Returns:
            Optional[int] : The id of the new pedestrian if it was created.
        """
        
        # Check target ids exist
        existing_target_ids = { target["id"] for target in self.topography["targets"] }
        for target_id in target_ids:
            if target_id not in existing_target_ids:
                return
                
        # Load default pedestrian json configuration
        with files("scenario_customizer.default_scenario_elements") \
            .joinpath('pedestrian.json').open('r') as pedestrian_file:
            pedestrian = json.load(pedestrian_file)

        # Write new pedestrian's values
        pedestrian["attributes"]["id"] = self.new_id
        pedestrian["position"]["x"] = x
        pedestrian["position"]["y"] = y
        pedestrian["targetIds"].extend(target_ids)
        

        self.topography["dynamicElements"].append(pedestrian)
        
        return pedestrian["attributes"]["id"]
