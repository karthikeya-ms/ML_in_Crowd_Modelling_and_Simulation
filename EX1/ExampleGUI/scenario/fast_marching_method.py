"""
This module contains our implementation of the fast marching method and any
helper classes or constants used.
"""

import math
from enum import Enum
from dataclasses import dataclass
from queue import PriorityQueue
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shared_types import Location

INF = float("inf")


class _CellStatus(str, Enum):
    """
    The cell statuses used in fast marching algorithm.
    """

    UNKNONW = "unknown"
    """
    The time to reach this cell is unknown.
    """
    NARROW_BAND = "narrow band"
    """
    This cell is a neighbor to another cell we know the time of.
    """
    FROZEN = "frozen"
    """
    The time to reach this cell is known.
    """


@dataclass(kw_only=True, order=True)
class FMMCell:
    """
    A class containing the information needed for fast marching per each cell in the grid.

    Attributes:
    -----------
    T: the time taken to reach this cell from the nearest target. Time here is just an integer, there is no particular unit.
    location: a tuple of (x,y) location of the cell. x is horizontal, y is vertical.
    status: The cell status of the fast marching algorithm.
    """

    T: float
    location: Location
    status: _CellStatus


class FastMarchingMethod:
    """
    An implementation of the fast marching method (FMM) for the shortest distance

    Attributes:
    -----------
    height: height of the grid.
    wifth: width of the grid.
    grid: a 2d list of size height x width. Each cell contains an instances of FMMCell.

    Private Attributes (use inside class):
    --------------------------------------
    _min_q : PriorityQueue[FMMCell]
        A priority queue used inside the fast marching algorithm to get the cell with the least T (time).
    """

    def __init__(
        self,
        width: int,
        height: int,
        targets: set[tuple[int, int]],
        obstacles: set[tuple[int, int]],
        measure_points: set[tuple[int, int]],
    ):
        """
        Creates an instance of the FastMarchingMethod, and runs the algorithm on the given grid configuration.

        Args:
            width (int): width of the grid.
            height (int): height of the grid.
            targets (set[tuple[int, int]]): a set containing the locations of the targets in the grid as (x,y) tuples.
            obstacles (set[tuple[int, int]]): a set containing the locations of the obstacles in the grid as (x,y) tuples.
            measure_points (set[tuple[int, int]]): a set containing the locations of the measure_points in the grid as (x,y) tuples.

        """
        self.grid: list[list[FMMCell]] = [
            [
                FMMCell(T=INF, location=(i, j), status=_CellStatus.UNKNONW)
                for j in range(width)
            ]
            for i in range(height)
        ]
        self.height = height
        self.width = width
        self._min_q: PriorityQueue[FMMCell] = PriorityQueue()

        for x, y in targets:
            i = y
            j = x
            self.grid[i][j].status = _CellStatus.NARROW_BAND
            self.grid[i][j].T = 0.0
            self._min_q.put(self.grid[i][j])

        for x, y in obstacles:
            i = y
            j = x
            self.grid[i][j].status = _CellStatus.FROZEN

        self._run()

    def _run(self) -> None:
        """
        Runs the fast marching algorithm on the given grid. Calculates the minimum time needed
        to reach each cell from the nearest target. Updates the grid property.
        """
        while not self._min_q.empty():
            A = self._min_q.get()
            A.status = _CellStatus.FROZEN
            for x, node in self._get_adjacent_nodes(A.location).items():
                if x in ["nw", "ne", "sw", "se"]:
                    continue
                if node.T == INF and not node.status == _CellStatus.FROZEN:
                    node.status = _CellStatus.NARROW_BAND
                    self._min_q.put(node)
                    node.T = self._solve_eikonal(node)

    def _get_adjacent_nodes(self, location: Location) -> dict[str, FMMCell]:
        """
        dict[str, FMMCell] : a dictionary containing the FMMCell of the neighboring 8 cells,
        ["nw", "n", "ne", "w", "e", "sw", "s", "se"]

        Gets the 8 cells adjacent to the given location on the grid.

        Args:
            location (tuple[int,int]): the location of the cell as tuple (i, j) where i is the row and j is the column.
        """
        i = location[0]
        j = location[1]
        directions = ["nw", "n", "ne", "w", "e", "sw", "s", "se"]
        neighbors = [
            self.grid[i + off_i][j + off_j]
            if 0 <= i + off_i < self.height and 0 <= j + off_j < self.width
            else None
            for off_i in [-1, 0, 1]
            for off_j in [-1, 0, 1]
            if abs(off_i) + abs(off_j) > 0
        ]
        return {
            direction: neighbor
            for direction, neighbor in zip(directions, neighbors, strict=True)
            if neighbor is not None
        }

    def _solve_eikonal(self, node: FMMCell) -> float:
        """
        float : the shortest time taken to reach the given cell from the nearest target

        The curve representing the time taken to reach closest target can be thought of
        as a convex function where the global minimum is at the target. The values of
        that functions are evaluated here using the speed F and the adjacent cell.

        Args:
            node (FMMCell): the node we want to calculate the time/cost for.
        """
        F = 1
        adjacent = self._get_adjacent_nodes(node.location)
        known_adjacent = {
            direction: node
            for direction, node in adjacent.items()
            if node.status == _CellStatus.FROZEN and node.T != INF
        }

        row_term = known_adjacent.get("w") or known_adjacent.get("e")
        if "w" in known_adjacent and "e" in known_adjacent:
            row_term = min(known_adjacent["w"], known_adjacent["e"])

        col_term = known_adjacent.get("n") or known_adjacent.get("s")
        if "n" in known_adjacent and "s" in known_adjacent:
            col_term = min(known_adjacent["n"], known_adjacent["s"])

        assert (
            row_term is not None or col_term is not None
        ), "Grid should have more than 1 cell"

        a = 2
        b = 0
        c = 0
        if row_term is not None:
            b += row_term.T
            c = row_term.T**2
        else:
            a -= 1.0

        if col_term is not None:
            b += col_term.T
            c = col_term.T**2
        else:
            a -= 1.0

        b = -2.0 * b
        c -= 1.0 / (F * F)
        d = (b**2) - (4 * a * c)
        sol1 = (-b - math.sqrt(d)) / (2 * a)
        sol2 = (-b + math.sqrt(d)) / (2 * a)
        return max(sol1, sol2)

    def plot_t_grid(self) -> None:
        """
        Plots the time/cost grid as a heatmap. useful for visualizing what the algorithm is doing.
        """
        t_grid = np.array([[cell.T for cell in row] for row in self.grid])
        t_grid[t_grid == INF] = np.nan
        plt.figure(figsize=(self.width, self.height))
        sns.heatmap(data=t_grid, annot=True)
        plt.show()

    def get_next_move(self, location: Location) -> tuple[int, int] | None:
        """
        tuple[int, int] | None : the location of the next cell according to the fast marching method.
        returns None if there are no valid moves to make.

        Uses the calculated time/cost grid in order to get the location of the best move from
        the given location.

        Args:
            location (tuple[int,int]): the location of the cell as tuple (x, y) where y is the row and x is the column.
        """

        x = location[0]
        y = location[1]
        assert (
            0 <= x < self.width and 0 <= y < self.height
        ), f"Cell coordinates ({x}, {y}) are outside of the grid of sise {self.width}x{self.height} (0-indexed)"
        current_node = self.grid[y][x]
        if current_node.T == INF or current_node.T == 0:
            return None

        adjacent = self._get_adjacent_nodes(current_node.location)
        minimum_node = min(adjacent.values(), key=lambda node: node.T)
        return (minimum_node.location[1], minimum_node.location[0])
