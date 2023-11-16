"""
This module defines the main method of the tool and some helper methods.
This method contains the main flow of the tool from parsing arguments to 
the interactive choice menu and their coordination.
"""
import argparse

from scenario_customizer import EXIT_KEY, HOME_KEY
from scenario_customizer.scenario import Scenario
from scenario_customizer.argument_parsing import parse_filename, parse_output, parse_pedestrians
from scenario_customizer.cli_input import string_input, scenario_input, output_file_path_input


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

def apply_arguments(args: argparse.ArgumentParser, scenario: Scenario) -> None:
    """Applies command line argument specified changes to the scenario.

    Args:
        args (argparse.ArgumentParser): The container for the command line arguments.
        scenario (Scenario): The scenario in which the changes will be applied.
    """
    if args.pedestrians is not None:
        for p in args.pedestrians:
            print(f"Adding pedestrian at ({p[0]},{p[1]}) with targets {p[2]}")
            scenario.add_pedestrian(p[0], p[1], p[2])

    scenario.save()

def main():
    """
    Contains the main tool execution flow.
    
    First arguments are parsed. Then the main scenario and save file parameters are obtained
    which is followed by applying the changes specyfied in the command line arguments and saving them.
    Finnaly, in interactive mode, the choice menu gives the user the oppurunity to dynamically customize
    the scenario.
    """
    parser = argparse.ArgumentParser(
        prog="Vadere Scenario Customizer",
        description="A tool to programatically customize Vadere scenario files using the command line."
    )

    parser.add_argument(
        '-f', '--filename', 
        type=parse_filename,
        help="The path to the scenario file to edit."
    )

    parser.add_argument(
        '-o', '--output',
        type=parse_output,
        help="The path to the output file where the new scenario will be saved. To be implemented."
    )

    parser.add_argument(
        '-s',
        action='store_true',
        help="A flag that, when active, skips all interactivity. This means the tool "
             "will execute actions specified by command line arguments and then terminate."
    )

    parser.add_argument(
        '-p',
        dest='pedestrians',
        help="Add a pedestrian as a tuple. It must come in the format: x,y(,t)* "
              "where x and y are floats in the form 0.0 and each t "
              "is an integer corresponding to the id of a target. "
              "This option can be repeated any amount of times."
        ,
        action='append', type=parse_pedestrians
    )

    args = parser.parse_args()
    args.pedestrians = list(filter(lambda x: x is not None, args.pedestrians))

    scenario = args.filename
    output_file_path = args.output
    skip_interactivity = args.s

    if skip_interactivity and (scenario is None or output_file_path is None):
        print("Selected no interactivity without both scenario and output file specified. "
              "No actions were performed.")
        return

    try:

        if scenario is None:
            scenario = scenario_input()

        if output_file_path is None:
            output_file_path = output_file_path_input()

        scenario.output_file_path = output_file_path
        apply_arguments(args, scenario)

        if skip_interactivity:
            raise EOFError()

        print("Welcome to the scenario customizer! At any time you can use:\n"
             f"{EXIT_KEY} or ctrl+D to exit;\n"
             f"{HOME_KEY} or ctrl+C to return to the main menu;"
        )

        while True:
            choice = int(string_input(
                menu_prompt=("Please choose an option:\n"
                "   1. Add pedestrian."),
                input_regex=r"^[1]$",
                fail_regex_message="Invalid choice!"
            ))

            try:
                if choice == 1:
                    add_pedestrian(scenario)

                scenario.save()
            except KeyboardInterrupt:
                # Used to come back to main menu
                pass

    except (EOFError, KeyboardInterrupt):
        print("\nYour actions have been successfully performed and saved.")

if __name__ == "__main__":
    main()