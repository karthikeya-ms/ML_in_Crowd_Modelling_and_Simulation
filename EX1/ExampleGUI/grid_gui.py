import tkinter as tk

class GridGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Cellular Automaton GUI")

        
        # Constants for grid size and cell size
        self.GRID_SIZE = 5
        self.CELL_SIZE = 60
        
        # Create a canvas for our grid
        self.canvas = tk.Canvas(self.master, width=self.GRID_SIZE*self.CELL_SIZE,
                                height=self.GRID_SIZE*self.CELL_SIZE)
        self.canvas.pack()

        # Draw initial grid
        self.draw_grid()
        
        # Bind mouse click events to the canvas
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-2>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_click)

    def draw_grid(self):
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                x1, y1 = i*self.CELL_SIZE, j*self.CELL_SIZE
                x2, y2 = x1+self.CELL_SIZE, y1+self.CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

    def on_canvas_click(self, event):
    # Calculate which square has been clicked on
        row = event.y // self.CELL_SIZE
        col = event.x // self.CELL_SIZE
        
        if event.num == 1:  # Left click for pedestrian
            self.add_pedestrian(col, row)
        elif event.num == 3:  # Right click for target
            self.add_target(col, row)
    def on_key_press(self, event):
        if event.char == 'o':  # Press 'o' key to add obstacle
            self.add_obstacle()

    def add_pedestrian(self, x, y):
        center_x, center_y = (x + 0.5)*self.CELL_SIZE, (y + 0.5)*self.CELL_SIZE
        self.canvas.create_oval(center_x-3, center_y-3, center_x+3, center_y+3, fill="red")

    def add_target(self, x, y):
        center_x, center_y = (x + 0.5)*self.CELL_SIZE, (y + 0.5)*self.CELL_SIZE
        self.canvas.create_oval(center_x-3, center_y-3, center_x+3, center_y+3, fill="blue")

    def on_obstacle_click(self, event):
        row = event.y // self.CELL_SIZE
        col = event.x // self.CELL_SIZE
        self.add_obstacle(col, row)

    def add_obstacle(self, x, y):
        x1, y1 = x*self.CELL_SIZE, y*self.CELL_SIZE
        x2, y2 = x1+self.CELL_SIZE, y1+self.CELL_SIZE
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")

if __name__ == "__main__":
    root = tk.Tk()
    app = GridGUI(root)
    root.mainloop()

    