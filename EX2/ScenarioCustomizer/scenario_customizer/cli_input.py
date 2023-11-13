"""
This module provides modular functions to request input 
from the command line with some automatic sanitization.
"""
import re
from json import JSONDecodeError

from scenario_customizer.scenario import Scenario, InvalidScenarioError

CLI_PROMPT = '> '
EXIT_KEY = 'q'
HOME_KEY = 'b'

BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

def string_input(
    menu_prompt: str = "",
    input_regex: str = ".*",
    fail_regex_message: str = "") -> str:
    """
    Basic call to python input with customizable sanitization routine.
    
    Specify a menu prompt to rquest the input,
    an input regular expression that the input must macth at least once
    and an error message in case the regular expression matching fails.

    Args:
        menu_prompt (str): The message the user will see when requested for input.
        input_regex (str): The regular expression that the input must match.
        fail_regex_message (str): The error message printed if the regular expression matching fails.

    Raises:
        EOFError: Raised if the user inputs 'q' or EOF (usually ctrl+D) 

    Returns:
        str: The input string provided by the user.
    """
    
    while True:
        print("\n" + menu_prompt)
        user_input = input(CLI_PROMPT)

        if user_input == EXIT_KEY:
            raise EOFError()
        
        if user_input == HOME_KEY:
            raise KeyboardInterrupt()

        if re.search(input_regex, user_input) is None:
            print(fail_regex_message)
            continue

        return user_input

def scenario_input() -> Scenario:
    """
    Wrapper method to string_input to request scenarios.

    Requests a scenario from the user and ensures it is valid.

    Returns:
        Scenario: The inputed scenario.
    """
    scenario = None
    while scenario is None:
        try:
            # matches anything, however ensures 'q' still exits (raises EOF)
            scenario_path = string_input(
                menu_prompt=("Insert the "
                            f"{UNDERLINE}{BOLD}path to the scenario file{END}{END}"
                             " you wish to costumize:"
                            )
            )

            scenario = Scenario(scenario_path)
        except FileNotFoundError:
            print(f"The file '{scenario_path}' was not found!")
        except JSONDecodeError:
            print(f"The file '{scenario_path}' is not valid json!")
        except InvalidScenarioError as e:
            print(e.error_message)

    return scenario

def output_file_path_input() -> str:
    """Wrapper method for string_input to request output file path.

    Returns:
        str: The inputed output file path.
    """
    while True:
        try:
            output_file_path = string_input(
                menu_prompt=( "Insert the "
                             f"{UNDERLINE}{BOLD}path to the save file{END}{END},"
                              " in which to save the new scenario:"
                            ),
                input_regex=r"[0-9a-zA-Z_][^/\\]*\.scenario$",
                fail_regex_message=("Invalid path. The new scenario file must end in"
                                    " .scenario and have a nonempty name.")
            )

            with open(output_file_path, 'w', encoding='utf-8'):
                pass

            return output_file_path
        except FileNotFoundError:
            print("The provided path is invalid!")