# Cellular Automata

Simple application for the cellular automaton in exercise 1.

# How to run

Change diectory to the `CellularAutomata` folder.
In there install all the dependencies (python 3.10+):
```bash
pip install -r requirements.txt
```

Finally, start the automata with:
```bash
python main.py
```

# Json Format for each Simulation Scenario

The automata uses the json format to store scenarios. This section documents the format of this file. An important point to note is that the `"measure_points"` property is optional.
```json
{
    "size": {
        "width": {
                "type": "int",
                "description": "The width of the grid in number of cells."
            },
            "height": {
                "type": "int",
                "description": "The height of the grid in number of cells."
            }
    },
    "pedestrians": [
        {
            "x": {
                "type": "int",
                "description": "The x coordinate of the pedestrian on the simulation grid."
            },
            "y": {
                "type": "int",
                "description": "The y coordinate of the pedestrian on the simulation grid."
            },
            "speed": {
                "type": "float",
                "description": "The speed of the pedestrian in cell per time step."
            }
        }
    ],
    "measure_points": [
        {
            "x": {
                "type": "int",
                "description": "The x coordinate of the measuring point on the simulation grid."
            },
            "y": {
                "type": "int",
                "description": "The y coordinate of the measuring point on the simulation grid."
            },
            "width": {
                "type": "int",
                "description": "The width of the measuring point."
            },
            "height": {
                "type": "int",
                "description": "The height of the measuring point."
            }
        }
    ],
    "targets": [
        {
            "x": {
                "type": "int",
                "description": "The x coordinate of the target on the simulation grid."
            },
            "y": {
                "type": "int",
                "description": "The y coordinate of the target on the simulation grid."
            }
        }
    ],
    "obstacles": [
        {
            "x": {
                "type": "int",
                "description": "The x coordinate of the obstacle on the simulation grid."
            },
            "y": {
                "type": "int",
                "description": "The y coordinate of the obstacle on the simulation grid."
            }
        }
    ]
}
```

