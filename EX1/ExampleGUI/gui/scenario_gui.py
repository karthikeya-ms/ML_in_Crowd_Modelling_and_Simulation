from __future__ import annotations

import time
import sys
import tkinter as tk
import threading as th

from scenario.scenario_elements import Scenario, Pedestrian
from gui.gui_callback import GuiCallback

class LeftMouseButton:
    """
    Helper class for the ScenarioGUI class. Contains basic logic related to click/hold/drag detction.
    These three types of actions refer to:
        click:
            Quick press and release of the mouse.
        hold:
            Press and release with some time in between, defined by the hold_seconds attribute.
        drag:
            Between pressing and releasing the mouse both some time passes and there is movement to different cells.

    Attributes:
    -----------
    is_dragging : bool
        Flag defining wether the mouse is being currently dragged (with the left mouse button pressed).


    Private Attributes (use inside class):
    --------------------------------------
    _gui : ScenarioGUI
        The gui object that uses this instance as a helper.
    _hold_seconds : float
        The number of seconds separating press and release that define a hold and a drag.
    _press_timestamp : Optional[float]
        The timestamp of when the current press started. It's set to None after release.
    _hold_event_id : Optional[str]
        The id, a string, of the current hold event. It's set to None after release.

    """

    def __init__(self, gui: ScenarioGUI, hold_seconds: float) -> None:
        """Creates an instance of the LeftMouseButton class.

        Args:
            gui (ScenarioGUI): The gui object that uses this instance as a helper.
            hold_seconds (float): The number of seconds separating press and release that define a hold and a drag. Defined by the gui.
        """
        self._gui = gui
        self._hold_seconds = hold_seconds

        self.is_dragging = False

        self._press_timestamp = None
        self._hold_event_id = None

    def press(self, event: tk.Event) -> None:
        """Handles all logic related to a left press.

        Args:
            event (tkinter.Event): The event object of the event that the press generated.
        """
        self._press_timestamp = time.time()
        self._hold_event_id = self._gui.master.after(
            int(1000 * ScenarioGUI.OBSTACLE_HOLD_SECONDS), self.drag, event
        )

    def drag(self, event: tk.Event) -> None:
        """
            Handles all logic related to a drag event. As this part of the logic is identical
            to a hold this method also handles hold calls (set by a call to after).

        Args:
            event (tkinter.Event): The event object of the event that the drag (hold) generated.
        """
        self.is_dragging = True
        self._gui.on_hold_or_drag(event)

    def release_was_click(self) -> bool:
        """Handles all release logic along with identifying if said release was a click.

        Returns:
            bool: If the release was a click returns True. It returns False otherwise.
        """
        elapsed_time = time.time() - self._press_timestamp
        self._gui.master.after_cancel(self._hold_event_id)

        self.is_dragging = False
        self._press_timestamp = None
        self._hold_event_id = None

        return elapsed_time < self._hold_seconds


class ScenarioGUI:
    """
        Handles visualization of a simulation scenario. It also handles user interaction with the scenario through the GUI,
        like placing elements with the mouse.

    Attributes:
    -----------
    master : tkinter.Frame
        The master frame of this gui. This public version of the property does not provide a setter.
    scenario : tkinter.Scenario
        The scenario to be rendered. This attribute's setter ensures that new scenarios are automatically rendered.

    Private Attributes (use inside class):
    -------------------------------
    _master : tkinter.Frame
        The tinter frame in which the scenario will be rendered.
    _canvas : tkinter.Canvas
        The canvas used to draw the scenario.
    _scenario : scenario_elements.Scenario
        The actual scenario to be rendered
    _grid_side : int
        The scenario's side length in number of cells.
    _cell_side : float
        A cell's side length in screen units.
    _canvas_side : float
        The canvas' side length in screen units.
    _element_radius : float
        The radius in screen units of a circular element, like pedestrians or targets.
    _obstacle_side : float
        The side of an obstacle in screen units.
    _grid_mode : bool
        If set to False the elements will be rendered on a uniform background (no visible cells).
    _heatmap_mode : bool
        If set to True the cells will be rendered in heatmap mode (takes priority over grid_mode).
    _left_mouse_button : LeftMouseButton
        Helper object to handle left mouse actions.

    """

    BACKGROUND_COLOR = "#EAEAEA"
    SEPARATOR_COLOR = "#FFFFFF"
    PEDESTRIAN_COLOR = "#FF0000"
    TARGET_COLOR = "#0000FF"
    OBSTACLE_COLOR = "#FF00FE"

    MIN_ELEMENT_RADIUS = 3
    MIN_OBSTACLE_SIDE = 3

    ELEMENT_TAG = "element"
    HEATMAP_TAG = "heatmap"

    OBSTACLE_HOLD_SECONDS = 0.5

    def __init__(
        self,
        master: tk.Frame,
        scen: Scenario,
        scenario_lock: th.Lock,
        max_canvas_dimensions: tuple[int, int] =(600, 600),
        grid_mode: bool = True,
        heatmap_mode: bool = False,
    ) -> None:
        """Creates an instance of the ScenarioGUI class. Also creates the canvas for the simulation and draws it.

        Args:
            master (tkinter.Frame): The tkinter frame object in which the grid will be rendered.
            scen (Scenario): The initial scenario to be rendered in the grid.
            grid_mode (bool, optional): Initial value of the grid_mode property. Defaults to True.
            heatmap_mode (bool, optional): Initial value of the heatmap_mode property. Defaults to False.
        """
        self._master: tk.Frame = master

        self._grid_mode: bool = grid_mode
        self._heatmap_mode: bool = heatmap_mode
        
        self._scenario: Scenario = scen
        self._scenario_lock = scenario_lock
        self._max_canvas_dimensions: tuple[int, int] = max_canvas_dimensions
        self._set_canvas_dimensions()
        
        self._left_mouse_button: LeftMouseButton = LeftMouseButton(
            self, ScenarioGUI.OBSTACLE_HOLD_SECONDS
        )

        #creating a canvas to be able to draw our grid
        self._canvas: tk.Canvas = tk.Canvas(
            self._master, width=self._canvas_width, height=self._canvas_height, highlightthickness=0
        )
        self._canvas.pack()

        # binding mouse click events to the canvas to interact
        self._canvas.bind("<Button-1>", GuiCallback((self._scenario_lock, ), self._on_left_press))
        self._canvas.bind("<ButtonRelease-1>", GuiCallback((self._scenario_lock, ), self._on_left_release))
        self._canvas.bind("<B1-Motion>", GuiCallback((self._scenario_lock, ), self._on_left_drag))
        if sys.platform == "darwin":
            self._canvas.bind("<Button-2>", GuiCallback((self._scenario_lock, ), self._on_right_click))
        else:
            self._canvas.bind("<Button-3>", GuiCallback((self._scenario_lock, ), self._on_right_click))

        # drawing the initial scenario
        self.draw_scenario()

    @property
    def master(self):
        """
        tkinter.Frame : The master frame of this gui.

        This public version of the property does not provide a setter. It is meant mainly for event scheduling.
        """
        return self._master

    @property
    def scenario(self) -> Scenario:
        """
        Scenario : The scenario to be rendered.

        This method just defines the "public" property for the scenario.
        """
        return self._scenario

    @scenario.setter
    def scenario(self, scen: Scenario) -> None:
        """
        void : The setter for the scenario property.

        This custom setter is the crux of the "public" scenario property.
        When updating the scenario the grid must be redrawn.
        This setter ensures these procedures are executed when the scenario is updated.
        """
        self._scenario = scen
        self._set_canvas_dimensions()
        self._canvas.configure(width=self._canvas_width, height=self._canvas_height)
        self.draw_scenario()

    @property
    def _grid_width(self) -> int:
        """
        int : The scenario's width length in number of cells.
        
        The number of cells the grid measures in width. Extracted dynamically from the scenario.
        """
        return self._scenario.width
    
    @property
    def _grid_height(self) -> int:
        """
        int : The scenario's height length in number of cells.
        
        The number of cells the grid measures in height. Extracted dynamically from the scenario.
        """
        return self._scenario.height
    

    @property
    def _element_radius(self) -> float:
        """
        float : The radius in screen units of a circular element, like pedestrians or targets.

        The element radius is, usually equal to cell_side*0.95/2.
        However when the grid has too many cells this radius generates invisible elements.
        As a compromise a minimum radius is used, meaning in large grids elements overflow from their cell.
        This lets the user visualize crowds in large grids at the cost of precision.
        """
        return max(ScenarioGUI.MIN_ELEMENT_RADIUS, self._cell_side * 0.95 / 2)

    @property
    def _obstacle_side(self) -> None:
        """
        float : The side of an obstacle in screen units.

        Obstacles usually fill the cell they are in.
        However when the grid has too many cells their size makes them invisible.
        As a compromise a minimum side length is used, meaning in large grids obstacles overflow from their cell.
        This lets the user visualize obstacles in large grids at the cost of precision.
        """
        return max(ScenarioGUI.MIN_OBSTACLE_SIDE, self._cell_side)

    @property
    def _max_canvas_width(self) -> float:
        """
        float : The max width of the canvas in screen units.

        The maximum width of the canvas that will render any scenario. 
        Used to calculate the actual width the canvas that will be used for any rendered scenario.
        """
        return self._max_canvas_dimensions[0]
    
    @property
    def _max_canvas_height(self) -> float:
        """
        float : The max height of the canvas in screen units.

        The maximum height of the canvas that will render any scenario. 
        Used to calculate the actual height the canvas that will be used for any rendered scenario.
        """
        return self._max_canvas_dimensions[1]

    def _set_canvas_dimensions(self):
        """
        This method uses the width and height bounds of the canvas 
        together with the width and height of the scenario (in number of cells)
        to be rendered to calculate the:
            Side of a cell
            Width and height of the canvas
        All in screen units.
        """
        self._cell_side = self._max_canvas_width / self._scenario.width
        
        if self._cell_side * self._scenario.height > self._max_canvas_height:
            self._cell_side = self._max_canvas_height / self._scenario.height
        
        # The plus one just ensures grid lines can be all drawn
        self._canvas_width = self._cell_side * self._scenario.width +1
        self._canvas_height = self._cell_side * self._scenario.height +1

    def draw_scenario(self) -> None:
        """
        Draws the grid from an empty canvas. This method should only be invoked when the grid itself changes,
        for example, when the number of cells is updated or the scenario itself is changed.
        However, when updating positions of any elements the method ScenarioGUI.update_grid is prefered.
        """

        self._canvas.delete('all')
        self._canvas.create_rectangle(
            0, 
            0, 
            self._canvas_width, 
            self._canvas_height, 
            fill=ScenarioGUI.BACKGROUND_COLOR
        )

        if self._heatmap_mode:
            self.draw_target_heatmap()
        elif self._grid_mode:
            self.draw_grid()

        self.update_scenario()

    def draw_grid(self) -> None:
        """Draws the grid."""
        for i in range(0, self._grid_width+1):
            x = i * self._cell_side
            self._canvas.create_line(
                x, 0, x, self._canvas_height, fill=ScenarioGUI.SEPARATOR_COLOR
            )
            
        for i in range(0, self._grid_height+1):
            y = i * self._cell_side
            self._canvas.create_line(
                0, y, self._canvas_width, y, fill=ScenarioGUI.SEPARATOR_COLOR
            )

    def draw_target_heatmap(self) -> None:
        """
        Draws the target heatmap on the canvas.
        This heatmap visualizes the distance from each cell to the nearest target.
        """

        def get_hex_number(target_distance: float, color_term: int) -> str:
            h = hex(max(0, min(255, int(10 * target_distance) - color_term * 255)))[2:]

            if len(h) == 1:
                h = "0" + h

            return h

        self._canvas.delete(ScenarioGUI.HEATMAP_TAG)
        for x in range(self._grid_width):
            for y in range(self._grid_height):
                target_distance = self.scenario.fast_marching.grid[y][x].T
                if target_distance == float("inf"):
                    target_distance = 0
                r = get_hex_number(target_distance, 0)
                g = get_hex_number(target_distance, 1)
                b = get_hex_number(target_distance, 2)
                self._canvas.create_rectangle(
                    x * self._cell_side,
                    y * self._cell_side,
                    (x + 1) * self._cell_side,
                    (y + 1) * self._cell_side,
                    fill=f"#{r}{g}{b}",
                    tag=ScenarioGUI.HEATMAP_TAG,
                )

    def update_scenario(self) -> None:
        """
        All elements on the grid are redrawn in their updated positions.
        This methods does not perform an update step in the scenario,
        it simply redraws every element with its current position.
        """
        self._canvas.delete(ScenarioGUI.ELEMENT_TAG)

        for pedestrian in self.scenario.pedestrians:
            x, y = pedestrian.position
            self._draw_pedestrian(x, y)

        for x, y in self.scenario.targets:
            self._draw_target(x, y)

        for x, y in self.scenario.obstacles:
            self._draw_obstacle(x, y)

    def _draw_pedestrian(self, x: int, y: int) -> None:
        """Uses the position of the pedestrian in the grid to draw it on the canvas.

        Args:
            x (int): The x axis value of the pedestrian's cell
            y (int): The y axis value of the pedestrian's cell
        """
        center_x, center_y = (x + 0.5) * self._cell_side, (y + 0.5) * self._cell_side
        self._canvas.create_oval(
            center_x - self._element_radius,
            center_y - self._element_radius,
            center_x + self._element_radius,
            center_y + self._element_radius,
            fill=ScenarioGUI.PEDESTRIAN_COLOR,
            tag=ScenarioGUI.ELEMENT_TAG,
        )

    def _draw_target(self, x: int, y: int) -> None:
        """Uses the position of the target in the grid to draw it on the canvas.

        Args:
            x (int): The x axis value of the target's cell
            y (int): The y axis value of the target's cell
        """
        center_x, center_y = (x + 0.5) * self._cell_side, (y + 0.5) * self._cell_side
        self._canvas.create_oval(
            center_x - self._element_radius,
            center_y - self._element_radius,
            center_x + self._element_radius,
            center_y + self._element_radius,
            fill=ScenarioGUI.TARGET_COLOR,
            tag=ScenarioGUI.ELEMENT_TAG,
        )

    def _draw_obstacle(self, x: int, y: int) -> None:
        """Uses the position of the obstacle in the grid to draw it on the canvas.

        Args:
            x (int): The x axis value of the obstacle's cell
            y (int): The y axis value of the obstacle's cell
        """
        self._canvas.create_rectangle(
            x * self._cell_side,
            y * self._cell_side,
            x * self._cell_side + self._obstacle_side,
            y * self._cell_side + self._obstacle_side,
            fill=ScenarioGUI.OBSTACLE_COLOR,
            tag=ScenarioGUI.ELEMENT_TAG,
        )

    def _on_right_click(self, event: tk.Event) -> None:
        """
            Handler for a right click event. Will calculate the cell clicked
            and add a target in it if possible (no other elements in it).

        Args:
            event (tkinter.Event): The tkinter event object.
        """
        # calculate which square has been clicked on
        pos = (int(event.x // self._cell_side), int(event.y // self._cell_side))
        if not (
            pos in self.scenario.pedestrians
            or pos in self.scenario.targets
            or pos in self.scenario.obstacles
        ):
            self.scenario.targets.add(pos)
            self.scenario.recompute_target_distances()
            if self._heatmap_mode:
                self.draw_scenario()
            else:
                self.update_scenario()

    def _on_left_press(self, event: tk.Event) -> None:
        """
            Handler for a left press event. It initializes all structures needed to differentiate
            between a left click, that adds a pedestrian, and a left hold/drag, that adds obstacles.

        Args:
            event (tkinter.Event): The tkinter event object.
        """
        self._left_mouse_button.press(event)

    def _on_left_drag(self, event: tk.Event) -> None:
        """
            Handler for a <B1-Motion> event which is triggered when the mouse moves while the left button is pressed.
            This handler detects wether the current invocation represents a drag.
            If it does then it adds an obstacle in the current cell if possible (no other elements in it).

        Args:
            event (tkinter.Event): The tkinter event object.
        """
        if not self._left_mouse_button.is_dragging:
            return

        self.on_hold_or_drag(event)

    def _on_left_release(self, event: tk.Event) -> None:
        """
            Handler for the release of the left mouse button.
            It will detect if this event represents a click.
            If it does it ads a pedestrian in the current cell (if possible).

        Args:
            event (tkinter.Event): The tkinter event object.
        """

        pos = (int(event.x // self._cell_side), int(event.y // self._cell_side))
        if self._left_mouse_button.release_was_click() and not (
            pos in self.scenario.pedestrians
            or pos in self.scenario.targets
            or pos in self.scenario.obstacles
        ):
            self.scenario.pedestrians.append(Pedestrian(pos, 1))
        else:
            # If the release doesn't represent a click than an obstacle was added
            # and the target distances must be recalculated
            self.scenario.recompute_target_distances()

        if self._heatmap_mode:
            self.draw_scenario()
        else:
            self.update_scenario()

    def on_hold_or_drag(self, event: tk.Event) -> None:
        """
            Handler for hold or drag "events". This handler will not be invoked directly by tkinter,
            but is, instead, a helper invoked every time a hold or a drag are detected to add an obstacle.

        Args:
            event (tkinter.Event): The tkinter event object.
        """
        obstacle = (int(event.x // self._cell_side), int(event.y // self._cell_side))
        if not (
            obstacle in self.scenario.targets or obstacle in self.scenario.pedestrians
        ):
            self.scenario.obstacles.add(obstacle)
            self.update_scenario()

    def activate_grid_mode(self) -> None:
        """The scenario will now be rendered in a normal background with a grid."""
        self._grid_mode = True
        self._heatmap_mode = False
        self.draw_scenario()

    def activate_heatmap_mode(self) -> None:
        """
        The scenario will now be rendered on a heatmap background in which 
        the temperature increases with the inverse of the distance to the target.
        This heatmap is drawn per cell on a grid.
        """
        self._heatmap_mode = True
        self.draw_scenario()

    def activate_free_range_mode(self) -> None:
        """The scenario will now be rendered in a normal backgorund with no gird."""
        self._grid_mode = False
        self._heatmap_mode = False
        self.draw_scenario()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Cellular Automaton GUI")
    frame = tk.Frame(root)
    scenario = Scenario(file_path="scenarios/test_scenario.json")
    app = ScenarioGUI(frame, scenario)
    frame.pack(pady=50, side=tk.TOP)
    root.mainloop()
