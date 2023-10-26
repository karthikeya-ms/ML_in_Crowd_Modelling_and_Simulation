from __future__ import annotations
from math import sqrt
from shared_types import Location, Color
from fast_marching_method import FastMarchingMethod
from numpy.typing import NDArray
import numpy as np
import json


class Scenario:
    """
    A scenario for a cellular automaton.
    """

    GRID_SIZE: Location = (500, 500)
    ID2NAME: dict[int, str] = {0: "EMPTY", 1: "TARGET", 2: "OBSTACLE", 3: "PEDESTRIAN"}
    NAME2ID: dict[str, int] = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3,
    }

    NAME2COLOR: dict[str, Color] = {
        "EMPTY": (255, 255, 255),
        "PEDESTRIAN": (255, 0, 0),
        "TARGET": (0, 0, 255),
        "OBSTACLE": (255, 0, 255),
    }

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

        self.grid_image = None
        self.pedestrians: list[Pedestrian] = []
        self.targets: set[Location] = set()
        self.obstacles: set[Location] = set()
        self.width = width
        self.height = height
        self.grid: NDArray[np.int8] = np.zeros((width, height)).astype(np.int8)

        for t in file_json["targets"]:
            self.grid[t["x"], t["y"]] = Scenario.NAME2ID["TARGET"]
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

    @position.setter
    def position(self, position: Location) -> None:
        self._position = position

    @property
    def desired_speed(self) -> float:
        return self._desired_speed

    @desired_speed.setter
    def desired_speed(self, speed: Location) -> None:
        self._desired_speed = speed

    @property
    def speed_offset(self) -> float:
        return self._speed_offset

    @speed_offset.setter
    def speed_offset(self, speed_offset: float) -> None:
        self._speed_offset = speed_offset

    def __hash__(self) -> int:
        return hash(self.position)

    def __eq__(self, other: Pedestrian | tuple[int, int]) -> bool:
        if isinstance(other, Pedestrian):
            return self.position == other.position
        elif isinstance(other, tuple):
            return self.position[0] == other[0] and self.position[1] == other[1]
        else:
            return False

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

            if next_pos in scenario.pedestrians and next_pos not in scenario.targets:
                self.speed_offset = 0.0
                break

            distance = sqrt(
                (self.position[0] - next_pos[0]) ** 2
                + (self.position[1] - next_pos[1]) ** 2
            )
            if distance > step_speed:
                break
            else:
                self.position = next_pos
                step_speed -= distance
        self.speed_offset = step_speed
