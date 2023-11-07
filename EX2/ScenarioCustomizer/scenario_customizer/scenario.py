import json
from typing import Optional

def assert_valid_scenario_recursive(properties_tree: dict, scenario_tree: dict) -> Optional[tuple[str, str]]:
    for prop in properties_tree:
        if prop not in scenario_tree:
            return f"The property '{prop}' was not found. Expected path: ", prop
        
        result = assert_valid_scenario_recursive(properties_tree[prop], scenario_tree[prop])
        
        if result is not None:
            return result[0], f"{prop}.{result[1]}"

    return None
    

class InvalidScenarioError(Exception):
    
    def __init__(self, error_cause_message: str, scenario_name: str =None, scenario_path: str =None) -> None:
        error_message = "Invalid scenario"
        
        if self.scenario_name is not None:
            error_message += f", named {self.scenario_name},"
        elif self.scenario_path is not None:
            error_message += f", in {self.scenario_path},"
        
        self.error_message = error_message + f" due to the following error:\n{error_cause_message}"
    
    def __str__(self) -> str:
        return self.error_message
            
        

class Scenario:
    
    required_property_tree = {
        "name": {},
        "scenario": {
            "topography": {
                "dynamicElemnts": {}
            }
        }
    }

    def __init__(self, path: str) -> None:
        self.scenario_path = path
        with open(path, 'r') as file:
            self.scenario_json = json.load(file)

        self.assert_valid_scenario()

    def assert_valid_scenario(self, path: str) -> None:
        result = assert_valid_scenario_recursive(Scenario.required_property_tree, self.scenario_json)
        
        if result is not None:
            error_cause_message = result[0] + result[1]
            if "name" in self.scenario_json:
                raise InvalidScenarioError(error_cause_message, scenario_name=self.scenario_json["name"])
            else:
                raise InvalidScenarioError(error_cause_message, scenario_path=self.scenario_path)
    
    

            
            
            
            

        
        
        
    
    
    
