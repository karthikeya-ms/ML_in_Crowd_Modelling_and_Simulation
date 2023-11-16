import sys
import re
from typing import Optional
from json import JSONDecodeError

from scenario_customizer.scenario import Scenario, InvalidScenarioError

def parse_filename(filename: str) -> Scenario:
    """Parses the scenario filename command line argument.

    Args:
        filename (str): The path to the scenario file passed by the user.

    Returns:
        Scenario: The loaded scenario file in the form of a scenario object.
    """
    try:
        return Scenario(filename)
    except (FileNotFoundError, JSONDecodeError, IsADirectoryError):
        print(f'The provided file path, {filename}, is invalid!')
    except InvalidScenarioError as e:
        print(e.error_message)

    sys.exit(1)

def parse_output(output: str) -> str:
    """Parses the output file command line argument.

    Args:
        output (str): The path in which the changes will be saved.

    Returns:
        str: The path inputed by the user if it is valid.
    """
    try:
        if re.search(r"[0-9a-zA-Z_][^/\\]*\.scenario$", output) is None:
            print('Output files must end with .scenario and contain a non-empty name.'
                  f'The provided path, {output}, is invalid.')
            sys.exit(1)

        with open(output, 'w', encoding='utf-8'):
            pass

        return output
    except (FileNotFoundError, IsADirectoryError):
        print(f'The provided file path, {output}, is invalid!')

    sys.exit(1)

def parse_pedestrians(pedestrian: str) -> Optional[list[tuple[float, float, list[int]]]]:
    """Checks the validity of the pedestrians inserted and casts the string to a usefull type.
    
    The return type of this cast is meant to be interpreted as a list of pedestrians. 
    In this list each pedestrian is represented by an x position, a float, a y position, a float, 
    and a list of target ids, a list of integers.
    
    Note that argsparse calls this function for every instance of the option called.

    Args:
        pedestrian (str): The pedestrian for adition inputed by the user.

    Returns:
        Optional[list[tuple[float, float, list[int]]]]: The list of pedestrians to add.
    """
    if re.search(r"^[0-9]+(\.[0-9]+)?,[0-9]+(\.[0-9]+)?(,[0-9]+)*$", pedestrian) is None:
        print(f'The following pedestrian was not inserted correctly: {pedestrian}. It will be skipped!')
        return None
    else:
        x, y, *target_ids = pedestrian.split(',')
        return (float(x), float(y), set(map(int, target_ids)))

