import sys
import tkinter
import time
import threading as t
from tkinter import Button, Canvas, Menu
from scenario_elements import Scenario, Pedestrian
from create_scenario import ScenarioCreator
from scenario_gui import ScenarioGUI
from scenario_loader import ScenarioLoader


class MainGUI:
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self):
        self._scenario = Scenario(file_path='scenarios/form_scenario_1.json')
        self.scenario_gui = None
        self.scenario_gui_mode = None

        self.is_playing = True
        self.play_thread = None


    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, scenario):
        self.scenario_gui.scenario = scenario
        self._scenario = scenario

    def create_scenario(
        self,
    ) -> None:
        ScenarioCreator()

    def restart_scenario(
        self,
    ) -> None:
        self.scenario = Scenario("scenarios/default.json")

    def play(
        self, button
    ):      
        button.config(text="Pause", command=lambda: self.pause(button))
        self.is_playing = True
        self.play_thread = t.Thread(target=self.play_loop)
        self.play_thread.start()

    def play_loop(self):
        while self.is_playing:
            time.sleep(0.5)
            self.step_scenario()

    def pause(
        self, button
    ):
        button.config(text="Play", command=lambda: self.play(button))
        self.is_playing = False
        self.play_thread = None

    def change_scen_gui_mode(
        self, *args
    ):
        if self.scenario_gui_mode.get() == 'Free Range':
            self.scenario_gui.activate_free_range_mode()
        elif self.scenario_gui_mode.get() == 'Grid':
            self.scenario_gui.activate_grid_mode()
        elif self.scenario_gui_mode.get() == 'Heatmap':
            self.scenario_gui.activate_heatmap_mode()

    def load_simulation(self):
        ScenarioLoader(self)

    def save_scenario(self):
        print('Save scenario not implemented yet ')

    def step_scenario(self):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        self.scenario.update_step()
        self.scenario_gui.update_scenario()

    def exit_gui(
        self,
    ) -> None:
        """
        Close the GUI.
        """
        sys.exit()

    def start_gui(
        self,
    ) -> None:
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
        self.scenario_gui = ScenarioGUI(grid_frame, self.scenario)

        top_bar = tkinter.Frame(win, height=50, width=1000)

        btn = Button(top_bar, text="Step Simulation", command=self.step_scenario)
        btn.grid(row=0, column=0, sticky='nswe')
        btn = Button(top_bar, text="Restart Simulation", command=self.restart_scenario)
        btn.grid(row=0, column=1, sticky='nswe')
        btn = Button(top_bar, text="Create Simulation", command=self.create_scenario)
        btn.grid(row=0, column=2, sticky='nswe')
        btn = Button(
            top_bar,
            text="Load Simulation",
            command=self.load_simulation,
        )
        btn.grid(row=0, column=3, sticky='nswe')

        self.scenario_gui_mode = tkinter.StringVar(top_bar, 'Grid')
        self.scenario_gui_mode.trace('w', self.change_scen_gui_mode)

        dropdown_label = tkinter.Label(top_bar, text='Mode Selector')
        dropdown_label.grid(row=1, column=0, sticky='nswe')

        dropdown = tkinter.OptionMenu(top_bar, self.scenario_gui_mode, 'Free Range', 'Grid', 'Heatmap')
        dropdown.grid(row=1, column=1, sticky='nswe')



        btn_play = Button(top_bar, text="Play", command=lambda: self.play(btn_play))
        btn_play.grid(row=1, column=2, sticky='nswe')

        btn = Button(top_bar, text="Save Scenario", command=self.save_scenario)
        btn.grid(row=1, column=3, sticky='nswe')

        top_bar.pack(side=tkinter.TOP)
        grid_frame.pack(side=tkinter.TOP)
        win.mainloop()
