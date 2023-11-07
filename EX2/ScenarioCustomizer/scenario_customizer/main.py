import argparse
import sys
from json import JSONDecodeError

from scenario_customizer.scenario import Scenario


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
        except EOFError:
            sys.exit(0)

def main():
    
    parser = argparse.ArgumentParser(
        prog="Vadere Scenario Customizer",
        description="A tool to programatically customize Vadere scenario files using the command line."
    )
    
    parser.add_argument(
        '-f', '--filename', 
        help="The path to the scenario file to edit."
    )
    
    args = parser.parse_args()
    
    if args.filename is not None:
        try:
            scenario = Scenario(args.filename)
        except FileNotFoundError or JSONDecodeError:
            print(f'The provided file path, {args.filname}, is invalid!')
            sys.exit(1)
    else:
        scenario = read_scenario_file()
    
    
    
    
    


if __name__ == "__main__":
    main()