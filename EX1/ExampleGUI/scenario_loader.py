import os
import tkinter

from scenario_elements import Scenario


class ScenarioLoader:
    
    def __init__(self, gui):

        self.gui = gui

        scenario_selector = tkinter.Tk()
        scenario_selector.title("Simulation Selector")

        self.window = scenario_selector

        main_frame = tkinter.Frame(scenario_selector)
        main_frame.pack(fill="both", expand=True)

        canvas = tkinter.Canvas(main_frame)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tkinter.ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y", expand=0)

        canvas.configure(yscrollcommand=scrollbar.set)

        label = tkinter.Label(canvas, text="Avalable scenarios:", font=("Helvetica 17 bold"))
        label.place(x=5, y=5)

        i = 0
        for scenario_file in os.listdir("./scenarios"):
            if scenario_file.endswith(".json"):
                btn = tkinter.Button(canvas, text=scenario_file[:-5], command=lambda path=f"scenarios/{scenario_file}": self.load_scenario(path))
                btn.place(x=15, y=i*30 + 50)
                i += 1
    
    def load_scenario(self, path):
        new_scenario = Scenario(path)
        self.gui.scenario = new_scenario

        self.window.destroy()


    