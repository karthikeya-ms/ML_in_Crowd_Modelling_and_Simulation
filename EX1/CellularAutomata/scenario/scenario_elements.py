from __future__ import annotations
import json
import numpy as np
from math import sqrt
from dataclasses import dataclass
from numpy.typing import NDArray
from shared_types import Location, Color

from scenario.fast_marching_method import FastMarchingMethod


class Scenario:
    """
    A scenario for a cellular automaton.
    """

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
        self.measure_points: list[MeasuringPoint] = []
        self.width = width
        self.height = height

        for t in file_json["targets"]:
            self.targets.add((t["x"], t["y"]))

        for p in file_json["pedestrians"]:
            self.pedestrians.append(Pedestrian((p["x"], p["y"]), p["speed"]))

        for o in file_json["obstacles"]:
            self.obstacles.add((o["x"], o["y"]))

        if "measure_points" in file_json:
            for m in file_json["measure_points"]:
                
                # ignore invalid measuring points
                if m["x"] + m["width"] > self.width \
                or m["y"] + m["height"] > self.height \
                or m["x"] + m["width"] < 0 \
                or m["y"] + m["height"] < 0:
                    continue
                
                self.measure_points.append(MeasuringPoint(m["x"], m["y"], m["width"], m["height"]))

        self.fast_marching = FastMarchingMethod(
            self.width, self.height, targets=self.targets, obstacles=self.obstacles)

        self.recompute_target_distances()

    def recompute_target_distances(self) -> None:
        self.fast_marching = FastMarchingMethod(
            self.width, self.height, targets=self.targets, obstacles=self.obstacles)

    def update_step(self) -> None:
        """
        Updates the position of all pedestrians.
        This does not take other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        for pedestrian in self.pedestrians:
            pedestrian.update_step(self)
        
        for measuring_point in self.measure_points:
            measuring_point.calculate_information(self)


@dataclass(kw_only=True)
class MeasuringPointInfo:
    pedestrian_count: int
    average_speed: float

class MeasuringPoint:
    
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.history = []

    def calculate_information(self, scenario: Scenario):
        inside = lambda p: self.x <= p.position[0] < self.x + self.width and self.y <= p.position[1] <  self.y + self.height
        pedestrians_in: list[Pedestrian] = list(filter(inside, scenario.pedestrians))
        number_of_pedestrians = len(pedestrians_in)
        speed_sum = sum(list(map(lambda p: p.desired_speed, pedestrians_in)))
        speed_avg = speed_sum / number_of_pedestrians if number_of_pedestrians > 0 else 0
        print(f"[second {len(self.history)}] Measuring point ({self.x},{self.y}) recorded: {number_of_pedestrians} pedestrians, average speed of {speed_avg} m/s")

        current_info = MeasuringPointInfo(pedestrian_count=number_of_pedestrians, average_speed=speed_avg)
        self.history.append(current_info)
        return current_info


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
