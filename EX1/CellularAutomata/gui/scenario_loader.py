import os
import tkinter


from scenario.scenario_elements import Scenario
from gui.create_scenario import handle_width_resize

class ScenarioLoader:
    """
    This class handles the loading of the scenario loader window.
    In this window the user can select the scenarios they want to loaf
    """

    def __init__(self, gui):
        """
            Upon construction, the ScenarioLoader creates a window that lists
            the available scenarios inside the scenarios directory.

        Args:
            gui (MainGUI): The GUI instance calling the scenario loader
        """

        self.main_gui = gui

        scenario_selector = tkinter.Tk()
        scenario_selector.geometry("500x500")
        scenario_selector.title("Simulation Loader")

        self.window = scenario_selector

        main_frame = tkinter.Frame(scenario_selector)
        main_frame.pack(fill="both", expand=True)

        canvas = tkinter.Canvas(main_frame)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tkinter.ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y", expand=0)

        canvas.configure(yscrollcommand=scrollbar.set)

        content_frame = tkinter.Frame(canvas)
        content_frame.pack(fill="both", anchor="nw")

        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        
        canvas.create_window(
            (0, 0), window=content_frame, anchor="nw", tags=("content_frame",)
        )
        canvas.bind("<Configure>", handle_width_resize)

        label = tkinter.Label(content_frame, text="Available Simulations:", font="Helvetica 17 bold")
        label.pack()

        i = 0
        for scenario_file in os.listdir("./scenarios"):
            if scenario_file.endswith(".json"):
                btn = tkinter.Button(content_frame, text=scenario_file[:-5], command=lambda path=f"scenarios/{scenario_file}": self.load_scenario(path))
                btn.pack()
                i += 1
    
    def load_scenario(self, path):
        """
            After selecting a scenario, this function updates the GUI with
            the new scenario. It then closes the scenario loader window.

        Args:
            path (str): The path to the scenario file
        """
        
        new_scenario = Scenario(path)
        self.main_gui.scenario = new_scenario
        self.main_gui.scenario_path = path

        self.window.destroy()


    