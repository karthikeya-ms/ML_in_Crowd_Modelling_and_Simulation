import sys
import tkinter
from tkinter import Button, Canvas, Menu
from scenario_elements import Scenario, Pedestrian
from create_scenario import ScenarioCreator
from grid_gui import ScenarioGUI
from scenario_loader import ScenarioLoader


class MainGUI:
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """
    
    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario):
        self.gui.scenario = scenario
        self._scenario = scenario

    def create_scenario(
        self,
    ):
        ScenarioCreator()

    def restart_scenario(
        self,
    ):
        print("restart not implemented yet")
    
    def play(
        self, button
    ):
        button.config(text="Pause", command=lambda: self.pause(button))
        print("play not implemented yet")
    
    def pause(
        self, button
    ):
        button.config(text="Play", command=lambda: self.play(button))
        print("pause not implemented yet")

    def load_simulation(self):
        ScenarioLoader(self)

    def step_scenario(self):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        self.scenario.update_step()
        self.gui.update_grid()

    def exit_gui(
        self,
    ):
        """
        Close the GUI.
        """
        sys.exit()

    def start_gui(
        self,
    ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry("700x700")  # setting the size of the window
        win.title("Cellular Automata GUI")

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label="Simulation", menu=file_menu)
        file_menu.add_command(label="New", command=self.create_scenario)
        file_menu.add_command(label="Restart", command=self.restart_scenario)
        file_menu.add_command(label="Close", command=self.exit_gui)

        grid_frame = tkinter.Frame(win, width=500, height=500)
        self._scenario = Scenario(50, 50, file_path='scenarios/test_scenario.json')
        self.gui = ScenarioGUI(grid_frame, self.scenario, grid_mode=True)
        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)

        top_bar = tkinter.Frame(win, height=50, width=1000)

        btn = Button(top_bar, text="Step simulation", command=self.step_scenario)
        btn.grid(row=0, column=0)
        btn = Button(top_bar, text="Restart simulation", command=self.restart_scenario)
        btn.grid(row=0, column=1)
        btn = Button(top_bar, text="Create simulation", command=self.create_scenario)
        btn.grid(row=0, column=2)
        btn = Button(
            top_bar,
            text="Load simulation",
            command=self.load_simulation,
        )
        btn.grid(row=0, column=3)
        btn = Button(top_bar, text="Play", command=lambda: self.play(btn))
        btn.grid(row=0, column=4)

        top_bar.pack(side=tkinter.TOP)
        grid_frame.pack(side=tkinter.TOP)
        win.mainloop()
