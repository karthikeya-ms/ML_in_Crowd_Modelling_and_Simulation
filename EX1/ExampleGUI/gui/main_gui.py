import sys
import tkinter
import time
import threading as t
from tkinter import Button, Canvas, Menu
from scenario.scenario_elements import Scenario, Pedestrian
from gui.create_scenario import ScenarioCreator
from gui.scenario_gui import ScenarioGUI
from gui.scenario_loader import ScenarioLoader
from gui.scenario_saver import ScenarioSaver
from gui.gui_callback import GuiCallback

class MainGUI:
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self):
        self._scenario = Scenario(file_path='scenarios/form_scenario_1.json')
        self._scenario_lock = t.Lock()
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
        button.config(text="Pause", command=GuiCallback((), lambda: self.pause(button)))
        self.is_playing = True
        self.play_thread = t.Thread(target=self.play_loop)
        self.play_thread.start()

    def play_loop(self):
        while self.is_playing:
            time.sleep(0.7)
            self._scenario_lock.acquire()
            self.step_scenario()
            self._scenario_lock.release()

    def pause(
        self, button
    ):
        button.config(text="Play", command=GuiCallback((), lambda: self.play(button)))
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

    def load_scenario(self):
        ScenarioLoader(self)

    def save_scenario(self):
        ScenarioSaver(self)

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
        file_menu.add_command(label="New", command=GuiCallback((self._scenario_lock,), self.create_scenario))
        file_menu.add_command(label="Restart", command=GuiCallback((self._scenario_lock,), self.restart_scenario))
        file_menu.add_command(label="Close", command=GuiCallback((self._scenario_lock,), self.exit_gui))

        grid_frame = tkinter.Frame(win, width=500, height=500)
        self.scenario_gui = ScenarioGUI(grid_frame, self.scenario, self._scenario_lock)

        top_bar = tkinter.Frame(win, height=50, width=1000)

        btn = Button(top_bar, text="Step Simulation", command=GuiCallback((self._scenario_lock,), self.step_scenario))
        btn.grid(row=0, column=0, sticky='nswe')
        btn = Button(top_bar, text="Restart Simulation", command=GuiCallback((self._scenario_lock,), self.restart_scenario))
        btn.grid(row=0, column=1, sticky='nswe')
        btn = Button(top_bar, text="Create Simulation", command=GuiCallback((self._scenario_lock,), self.create_scenario))
        btn.grid(row=0, column=2, sticky='nswe')
        btn = Button(
            top_bar,
            text="Load Simulation",
            command=GuiCallback((self._scenario_lock,), self.load_scenario),
        )
        btn.grid(row=0, column=3, sticky='nswe')

        self.scenario_gui_mode = tkinter.StringVar(top_bar, 'Grid')
        self.scenario_gui_mode.trace('w', GuiCallback((self._scenario_lock,), self.change_scen_gui_mode))

        dropdown_label = tkinter.Label(top_bar, text='Mode Selector')
        dropdown_label.grid(row=1, column=0, sticky='nswe')

        dropdown = tkinter.OptionMenu(top_bar, self.scenario_gui_mode, 'Free Range', 'Grid', 'Heatmap')
        dropdown.grid(row=1, column=1, sticky='nswe')



        btn_play = Button(top_bar, text="Play", command=GuiCallback((), lambda: self.play(btn_play)))
        btn_play.grid(row=1, column=2, sticky='nswe')

        btn = Button(top_bar, text="Save Simulation", command=GuiCallback((self._scenario_lock,), self.save_scenario))
        btn.grid(row=1, column=3, sticky='nswe')

        top_bar.pack(side=tkinter.TOP)
        grid_frame.pack(side=tkinter.TOP)
        win.mainloop()
