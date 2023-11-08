import argparse
import sys
from json import JSONDecodeError

from scenario_customizer.scenario import Scenario, InvalidScenarioError
from scenario_customizer.cli_input import cli_input


def read_scenario_file() -> Scenario:
    print("Welcome to the Scenario Customizer.")
    print("To get started insert the path to the Scenario file you wish to costumize: (q to exit)")
    scenario = None
    while scenario is None:
        try:
            scenario_path = input("> ")

            if scenario_path == "q":
                raise EOFError()

            scenario = Scenario(scenario_path)
        except FileNotFoundError:
            print(f"The file '{scenario_path}' was not found! (q to exit)")
        except JSONDecodeError:
            print(f"The file '{scenario_path}' is not valid json! (q to exit)")
        except InvalidScenarioError as e:
            print(e.error_message)

    return scenario

def add_pedestrian(scenario):
    print("oi")
    x, y = cli_input(
        "Please insert the position of the pedestrian following the pattern 'x,y':",
        r"^\d+([.]\d+)?,\d+([.]\d+)?$",
        "Invalid position! The position must follow the pattern 'x,y'."
    ).split(',')
    print("oi")

    x, y = float(x), float(y)

    print("Finnaly, please insert the target ids for this pedestrian: (s to stop)")
    target_ids = set()
    while True:
        new_id = cli_input(
            "Next id: (s to stop)",
            r"^(\d+|[s])$",
            "Invalid id!"
        )
        if new_id == "s":
            break

        target_ids.add(int(new_id))

    pedestrian_id = scenario.add_pedestrian(x, y, target_ids)
    if pedestrian_id is None:
        print("Could not add the pedestrian!")
    else:
        print(f"Pedestrian added with id {pedestrian_id}.")

def save_changes(scenario):
    while True:
        try:
            print("Insert the path in which to save the new scenario:")
            path = input('> ')
            if path == "q":
                raise EOFError()
            scenario.save(path)
            break
        except FileExistsError:
            print("Can't save changes to the same scenario file!")
        except FileNotFoundError:
            print("The provided path is invalid!")



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
        if args.filename is not None:
            try:
                scenario = Scenario(args.filename)
            except (FileNotFoundError, JSONDecodeError):
                print(f'The provided file path, {args.filname}, is invalid!')
                sys.exit(1)
            except InvalidScenarioError as e:
                print(e.error_message)
                sys.exit(1)
        else:
            scenario = read_scenario_file()

        while True:
            choice = int(cli_input(
                ("Please choose an option: (q to quit)\n"
                 "   1. Add pedestrian.\n"
                 "   2. Save changes."),
                r"^[1-2]$",
                "Invalid choice!"
            ))

            if choice == 1:
                add_pedestrian(scenario)
            else:
                save_changes(scenario)

    except (EOFError, KeyboardInterrupt):
        pass



if __name__ == "__main__":
    main()