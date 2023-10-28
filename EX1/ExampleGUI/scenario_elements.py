import scipy.spatial.distance
from PIL import Image, ImageTk
from math import sqrt
from shared_types import Location, Color
from fast_marching_method import FastMarchingMethod
from numpy.typing import NDArray
from tkinter import Canvas
import numpy as np
import json



class Scenario:
    """
    A scenario for a cellular automaton.
    """

    GRID_SIZE: Location = (500, 500)

    def get_neighbours(self, x_coord: int, y_coord: int) -> list[Location]:
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param x_coord: The x coordinate of the current position
        :param y_coord: The y coordinate of the current position
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (adjacent_x + x_coord, adjacent_y + y_coord)
            for adjacent_x in [-1, 0, 1]
            for adjacent_y in [-1, 0, 1]
            if 0 <= adjacent_x + x_coord < self.width
            and 0 <= adjacent_y + y_coord < self.height
            and np.abs(adjacent_x) + np.abs(adjacent_y) > 0
        ]

    def __init__(self, file_path: str):
        with open(file_path, "r") as file:
            file_json = json.load(file)
            width = file_json["size"]["width"]
            height = file_json["size"]["height"]

        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")
        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.pedestrians: list[Pedestrian] = []
        self.targets: set[Location] = set()
        self.obstacles: set[Location] = set()
        self.width = width
        self.height = height

        for t in file_json["targets"]:
            self.targets.add((t["x"], t["y"]))

        for p in file_json["pedestrians"]:
            self.pedestrians.append(Pedestrian((p["x"], p["y"]), p["speed"]))

        for o in file_json["obstacles"]:
            self.obstacles.add((o["x"], o["y"]))

        self.fast_marching = FastMarchingMethod(
            self.width, self.height, targets=self.targets, obstacles=self.obstacles
        )

        self.recompute_target_distances()

    def recompute_target_distances(self) -> None:
        self.fast_marching = FastMarchingMethod(
            self.width, self.height, targets=self.targets, obstacles=self.obstacles
        )

    def update_step(self) -> None:
        """
        Updates the position of all pedestrians.
        This does not take other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        for pedestrian in self.pedestrians:
            pedestrian.update_step(self)


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position: Location, desired_speed: float) -> None:
        self._position = position
        self._desired_speed = desired_speed
        self._speed_offset: float = 0.0

    @property
    def position(self) -> Location:
        return self._position

    @property
    def desired_speed(self) -> float:
        return self._desired_speed

    @property
    def speed_offset(self) -> float:
        return self._speed_offset

    @speed_offset.setter
    def speed_offset(self, speed_offset: float) -> None:
        self._speed_offset = speed_offset

    def get_neighbors(self, scenario: Scenario) -> list[Location]:
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return scenario.get_neighbours(self._position[0], self._position[1])

    def update_step(self, scenario: Scenario) -> None:
        """
        Moves to the cell with the lowest distance to the target.
        This does not take other pedestrians into account.
        Pedestrians can occupy the same cell.

        :param scenario: The current scenario instance.
        """
        step_speed = self.desired_speed + self.speed_offset
        while True:
            next_pos = scenario.fast_marching.get_next_move(self.position)
            if next_pos is None:
                break

            distance = sqrt(
                (self.position[0] - next_pos[0]) ** 2
                + (self.position[1] - next_pos[1]) ** 2
            )
            if distance > step_speed:
                break
            else:
                self._position = next_pos
                step_speed -= distance
        self.speed_offset = step_speed
