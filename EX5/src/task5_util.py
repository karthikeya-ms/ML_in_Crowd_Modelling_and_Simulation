"""This module contains multiple utilities used in task 5.
"""

import numpy as np
import pandas as pd
from math import sqrt
from typing import Iterable
from math import inf

def time_delay_embedding(data_points: pd.DataFrame, delay: int =350) -> np.ndarray:
    """Create time delay embedding using time series data.

    Generates a time delay embedding using given datapoints from a time series. 
    The resulting embedding will have a time delay as specified.

    Args:
        data_points (pd.DataFrame): The time series data used.
        delay (int, optional): The time delay value. Defaults to 350.

    Returns:
        np.ndarray: The time delay embedding.
    """


    n_rows, _ = data_points.shape
    assert n_rows > delay

    return data_points[:delay+1].values
    
def distance(x1: Iterable, x2: Iterable) -> float:
    """Calculates euclidean distance between two points of arbitrary dimentions.

    Args:
        x1 (Iterable): point 1
        x2 (Iterable): point 2

    Returns:
        float: distance
    """

    total = 0
    for e1, e2 in zip(x1,x2):
        total += (e1-e2)**2

    return sqrt(total)

def get_trajectory_distances(points: Iterable) -> np.ndarray:
    """Calculates the distances between conecutive pairs of points in a trajectory.

    Args:
        points (Iterable): The trajectory of points

    Returns:
        np.ndarray: The distances
    """

    distances = np.zeros((len(points)-1))

    for i in range(len(distances)):
        distances[i] = distance(points[i], points[i+1])

    return distances

def get_speed(speed_map: dict, current_arc_length: float) -> float:
    """Calculates the approximate speed for a certain arc length.

    Args:
        speed_map (dict): the mapping from known arc_lengths to speeds.
        current_arc_length (float): the current arc length.

    Returns:
        float: the known speed of the arc length closest to the given one.
    """
    
    
    closest_mapped_point = min(speed_map.keys(), key= lambda k: abs(current_arc_length-k))
    return speed_map[closest_mapped_point]

def get_y_value(x_value, c, l_vector, epsilon):
        y_value = 0
        for i in range(c.shape[0]):
            y_value += c[i] * np.exp(-((x_value - l_vector[i]) / epsilon) ** 2)
        return y_value