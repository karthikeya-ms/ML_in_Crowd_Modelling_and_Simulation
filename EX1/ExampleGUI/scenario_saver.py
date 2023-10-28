from __future__ import annotations
from typing import TYPE_CHECKING

import os.path
import tkinter


from scenario_elements import Scenario
import json


if TYPE_CHECKING:
    from gui import MainGUI



# Note: For cleaner code, normally we'd use json.JSONEncoder. But this is a simple enough case that we don't need to.
def scenario_to_json(scenario: Scenario) -> str:
    """
    Convert a scenario to a JSON string including indentation for readability.
    @param scenario: The scenario to convert.
    @return:         The JSON string.
    """
    scenario_as_dict = {
        "size": {
            "width": scenario.width,
            "height": scenario.height,
        },
        "pedestrians": [
            {
                "x": p.position[0],
                "y": p.position[1],
                "speed": p.desired_speed,
            }
            for p in scenario.pedestrians
        ],
        "targets": [
            {
                "x": t[0],
                "y": t[1],
            }
            for t in scenario.targets
        ],
        "obstacles": [
            {
                "x": o[0],
                "y": o[1],
            }
            for o in scenario.obstacles
        ],
    }
    return json.dumps(scenario_as_dict, indent=4)


class ScenarioSaver:
    def __init__(self, main_gui: MainGUI):
        """
        Initiate save scenario dialog
        @param main_gui: Main GUI instance
        """
        self.main_gui = main_gui

        self.window = tkinter.Tk()
        self.window.geometry("500x200")
        self.window.title("Save Scenario")

        label = tkinter.Label(self.window, text="Save scenario as:", font="Helvetica 17 bold")
        label.pack()

        filename_entry = tkinter.Entry(self.window, width=50)
        filename_entry.pack()

        save_button = tkinter.Button(self.window, text="Save", command=lambda: self.save_json(
            str(filename_entry.get()),
            scenario_to_json(self.main_gui.scenario)
        ))

        save_button.pack()

    def save_json(self, filename: str, json_content: str) -> None:
        """
        Attempt to save json-serialized scenario to user-specified file name. Closes the save dialog if successful.
        @param filename:     Desired filename save the .json ending. Must not be empty!
        @param json_content: Scenario as json string
        """
        # TODO later, UI notification message
        if not filename:
            print('Filename cannot be empty.')
            return

        if not filename.endswith('.json'):
            filename += '.json'

        filename = f'scenarios/{filename}'

        # TODO Later, yes/no dialog to confirm overwriting
        if os.path.exists(filename):
            print(f'File {filename} already exists. Overwriting.')

        print(f'Saving scenario as {filename}')

        try:
            with open(filename, 'w') as f:
                f.write(json_content)
        # TODO UI notification message
        except Exception as e:
            print(f'Error saving scenario: {e}')
            return

        self.window.destroy()
