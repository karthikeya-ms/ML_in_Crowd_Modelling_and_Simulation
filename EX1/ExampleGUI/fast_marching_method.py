from enum import Enum
from dataclasses import dataclass
from queue import PriorityQueue
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shared_types import Location

INF = float("inf")


class _CellStatus(str, Enum):
    UNKNONW = "unknown"
    NARROW_BAND = "narrow band"
    FROZEN = "frozen"


@dataclass(kw_only=True, order=True)
class FMMCell:
    T: float
    location: Location
    status: _CellStatus


class FastMarchingMethod:
    def __init__(
        self,
        width: int,
        height: int,
        targets: set[tuple[int, int]],
        obstacles: set[tuple[int, int]],
    ):
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
        t_grid = np.array([[cell.T for cell in row] for row in self.grid])
        t_grid[t_grid == INF] = np.nan
        plt.figure(figsize=(self.width, self.height))
        hm = sns.heatmap(data=t_grid, annot=True)
        plt.show()

    def get_next_move(self, location: Location) -> tuple[int, int] | None:
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
