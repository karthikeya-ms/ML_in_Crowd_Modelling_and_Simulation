import os
import tkinter


from scenario.scenario_elements import Scenario
from gui.create_scenario import handle_width_resize

class ScenarioLoader:
    
    def __init__(self, gui):

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
        new_scenario = Scenario(path)
        self.main_gui.scenario = new_scenario

        self.window.destroy()


    