import tkinter as tk
from tkinter import ttk
from enum import Enum
import json

ENTRY_TYPE = dict[str, int | float]


class FrameType(str, Enum):
    GRID_SIZE = "Grid Size"
    PEDESTRIANS = "Pedestrians"
    TARGETS = "Targets"
    OBSTACLES = "Obstacles"
    SCENARIO_INFO = "Scenario Info"


SINGLE_ROW_TYPES = (FrameType.GRID_SIZE, FrameType.SCENARIO_INFO)


class InputRow:
    def __init__(
        self,
        outer_frame: tk.Frame,
        row: int,
        input_field_names: list[str],
        field_width: int,
    ):
        self.entries: list[tk.Entry] = []
        for index, field_name in enumerate(input_field_names, 0):
            frame = tk.Frame(outer_frame, borderwidth=0, highlightthickness=0)
            frame.grid(row=row, column=index, sticky="nswe", padx=10)

            tk.Label(frame, text=f"{field_name}:", font=("Helvetica 17")).grid(
                row=0, column=0
            )
            entry = tk.Entry(frame, width=field_width)
            entry.grid(row=0, column=1)

            self.entries.append(entry)


class InputFrame:
    def __init__(self, outer_frame: tk.Frame, frame_type: FrameType):
        self.frame = tk.Frame(outer_frame, borderwidth=0, highlightthickness=0)
        self.frame.pack(fill="x")
        self.frame_type = frame_type
        self.input_rows: list[InputRow] = []
        self.next_row = 1

        tk.Label(
            self.frame, text=self.frame_type.value, font=("Helvetica 17 bold")
        ).grid(row=0, column=0, padx=5, pady=20, sticky="w")

        if self.frame_type not in SINGLE_ROW_TYPES:
            tk.Button(
                self.frame,
                text=f"Add {self.frame_type.value[:-1]}",
                width=8,
                command=self.add_input_row,
            ).grid(row=0, column=1, padx=5, pady=20)
        else:
            self.add_input_row()

    def add_input_row(self):
        input_field_names: list[str] = []
        match self.frame_type:
            case FrameType.GRID_SIZE:
                input_field_names = ["Width", "Height"]
            case FrameType.PEDESTRIANS:
                input_field_names = ["X", "Y", "Speed"]
            case FrameType.SCENARIO_INFO:
                input_field_names = ["Name"]
            case other:
                input_field_names = ["X", "Y"]

        self.input_rows.append(
            InputRow(
                outer_frame=self.frame,
                row=self.next_row,
                input_field_names=input_field_names,
                field_width=20 if self.frame_type == FrameType.SCENARIO_INFO else 5,
            )
        )

        self.next_row += 1


def handle_width_resize(event):
    canvas = event.widget
    canvas_frame = canvas.nametowidget(canvas.itemcget("content_frame", "window"))
    min_width = canvas_frame.winfo_reqwidth()
    if min_width < event.width:
        canvas.itemconfigure("content_frame", width=event.width)


class ScenarioCreator:
    def __init__(self):
        self.win = tk.Tk()
        self.win.geometry("500x500")
        self.win.title("Scenario Creator")

        main_frame = tk.Frame(self.win)
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y", expand=0)

        canvas.configure(yscrollcommand=scrollbar.set)

        self.content_frame = tk.Frame(canvas)
        self.content_frame.pack(fill="both", anchor="nw")

        self.content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw", tags=("content_frame",)
        )
        canvas.bind("<Configure>", handle_width_resize)

        self.grid_size_frame = InputFrame(
            outer_frame=self.content_frame, frame_type=FrameType.GRID_SIZE
        )

        ttk.Separator(self.content_frame, orient="horizontal").pack(fill="x", pady=5)

        self.pdestrians_frame = InputFrame(
            outer_frame=self.content_frame,
            frame_type=FrameType.PEDESTRIANS,
        )

        ttk.Separator(self.content_frame, orient="horizontal").pack(fill="x", pady=5)

        self.targets_frame = InputFrame(
            outer_frame=self.content_frame, frame_type=FrameType.TARGETS
        )

        ttk.Separator(self.content_frame, orient="horizontal").pack(fill="x", pady=5)

        self.obstacles_frame = InputFrame(
            outer_frame=self.content_frame, frame_type=FrameType.OBSTACLES
        )

        ttk.Separator(self.content_frame, orient="horizontal").pack(fill="x", pady=5)

        self.scenario_info_frame = InputFrame(
            outer_frame=self.content_frame, frame_type=FrameType.SCENARIO_INFO
        )

        tk.Button(
            self.content_frame,
            text=f"Save Scenario",
            width=8,
            command=self.save,
        ).pack(fill="x", padx=100)
        
        tk.Button(
            self.content_frame,
            text=f"Save Scenario and Close",
            width=8,
            command=self.save_and_close,
        ).pack(fill="x", padx=100)

    def save(self):
        size_dict: ENTRY_TYPE = {
            "width": int(self.grid_size_frame.input_rows[0].entries[0].get()),
            "height": int(self.grid_size_frame.input_rows[0].entries[1].get()),
        }

        pedestrian_data: list[ENTRY_TYPE] = [
            {
                "x": int(input_row.entries[0].get()),
                "y": int(input_row.entries[1].get()),
                "speed": float(input_row.entries[2].get()),
            }
            for input_row in self.pdestrians_frame.input_rows
        ]

        target_data: list[ENTRY_TYPE] = [
            {
                "x": int(input_row.entries[0].get()),
                "y": int(input_row.entries[1].get()),
            }
            for input_row in self.targets_frame.input_rows
        ]

        obstacle_data: list[ENTRY_TYPE] = [
            {
                "x": int(input_row.entries[0].get()),
                "y": int(input_row.entries[1].get()),
            }
            for input_row in self.obstacles_frame.input_rows
        ]

        scenario_dict: dict[str, list[ENTRY_TYPE] | ENTRY_TYPE] = {
            "size": size_dict,
            "pedestrians": pedestrian_data,
            "targets": target_data,
            "obstacles": obstacle_data,
        }

        scenario_name = self.scenario_info_frame.input_rows[0].entries[0].get()

        scenario_json = json.dumps(scenario_dict, indent=4)

        with open(f"scenarios/{scenario_name}.json", "w") as out:
            out.write(scenario_json)

    def save_and_close(self):
        self.save()
        self.win.destroy()