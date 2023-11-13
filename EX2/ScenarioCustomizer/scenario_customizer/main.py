import argparse
import sys
from json import JSONDecodeError

from scenario_customizer.scenario import Scenario, InvalidScenarioError
from scenario_customizer.cli_input import string_input, scenario_input, EXIT_KEY, HOME_KEY, output_file_path_input

def add_pedestrian(scenario: Scenario):
    """Requests the parameters from the user to create a new pedestrian.

    Args:
        scenario (Scenario): The scenatio in which to add the new pedestrian.
    """
    x, y = string_input(
        menu_prompt="Please insert the position of the pedestrian following the pattern 'x,y':",
        input_regex=r"^\d+([.]\d+)?,\d+([.]\d+)?$",
        fail_regex_message="Invalid position! The position must follow the pattern 'x,y'."
    ).split(',')

    x, y = float(x), float(y)

    print("Finnaly, please insert the target ids for this pedestrian: (s to stop)")
    target_ids = set()
    while True:
        new_id = string_input(
            menu_prompt="Next id: (s to stop)",
            input_regex=r"^(\d+|[s])$",
            fail_regex_message="Invalid id!"
        )
        if new_id == "s":
            break

        target_ids.add(int(new_id))

    pedestrian_id = scenario.add_pedestrian(x, y, target_ids)
    if pedestrian_id is None:
        print("Could not add the pedestrian! Targets added are not subset of available targets!")
    else:
        print(f"Pedestrian added with id {pedestrian_id}.")
        scenario.save()


def main():

    parser = argparse.ArgumentParser(
        prog="Vadere Scenario Customizer",
        description="A tool to programatically customize Vadere scenario files using the command line."
    )

    parser.add_argument(
        '-f', '--filename', 
        help="The path to the scenario file to edit."
    )

    parser.add_argument(
        '-p', '--add-pedestrian',
        help="Add a pedestrian as a position tuple x,y. To be implemented."
    )

    parser.add_argument(
        '-o', '--output',
        help="The path to the output file where the new scenario will be saved. To be implemented."
    )


    args = parser.parse_args()


    try:
        scenario = None
        output_file_path = None

        if args.filename is not None:
            try:
                scenario = Scenario(args.filename)
            except (FileNotFoundError, JSONDecodeError):
                print(f'The provided file path, {args.filname}, is invalid!')
                sys.exit(1)
            except InvalidScenarioError as e:
                print(e.error_message)
                sys.exit(1)

        if scenario is None:
            scenario = scenario_input()

        if output_file_path is None:
            output_file_path = output_file_path_input()

        print("Welcome to the scenario customizer! At any time you can use:\n"
             f"{EXIT_KEY} or ctrl+D to exit;\n"
             f"{HOME_KEY} or ctrl+C to return to the main menu;"
        )

        while True:
            choice = int(string_input(
                menu_prompt=("Please choose an option:\n"
                "   1. Add pedestrian.\n"
                "   2. Save changes."),
                input_regex=r"^[1]$",
                fail_regex_message="Invalid choice!"
            ))

            try:
                if choice == 1:
                    add_pedestrian(scenario)
            except KeyboardInterrupt:
                # Used to come back to main menu
                pass

    except (EOFError, KeyboardInterrupt):
        pass



if __name__ == "__main__":
    main()