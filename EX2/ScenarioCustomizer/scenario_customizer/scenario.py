import json

class Scenario:
    
    def __init__(self, path: str) -> None:
        with open(path, 'r') as file:
            self.scenario_json = json.load(file)
    
    
    
    
