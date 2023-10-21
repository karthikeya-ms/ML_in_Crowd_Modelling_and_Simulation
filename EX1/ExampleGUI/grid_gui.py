import time
import tkinter as tk

class GridGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Cellular Automaton GUI")

        #the constants for the grid size and sizes of the cells 
        self.GRID_SIZE = 10
        self.CELL_SIZE = 40

        #colours used for the different elements like pedestrians, targets, bg etc.
        self.BACKGROUND_COLOR = "e9ecef" #bg colour
        self.PEDESTRIAN_COLOR = "#e63946"  # red colour for pedestrian
        self.TARGET_COLOR = "#457b9d"   #  blue colour for target
        self.OBSTACLE_COLOR = "#2b2d42"  # Dark grayish colour for obstacle
        self.HOVER_COLOR = "#f8f9fa"  # A hover color

        #creating a canvas to be able to draw our grid
        self.canvas = tk.Canvas(self.master, width=self.GRID_SIZE*self.CELL_SIZE,
                                height=self.GRID_SIZE*self.CELL_SIZE)
        self.canvas.pack()
        self.canvas.create_rectangle(0, 0, self.GRID_SIZE*self.CELL_SIZE, self.GRID_SIZE*self.CELL_SIZE, fill="#e9ecef", outline="")
        self.canvas.create_rectangle(0, 0, self.GRID_SIZE*self.CELL_SIZE, self.GRID_SIZE*self.CELL_SIZE/2, fill="#f8f9fa", outline="", stipple='gray12')

        #drawing the initial grid
        self.draw_grid()
        
        #binding mouse click events to the canvas to interact
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_hover)
        
    #drawing the initial grid on the canvas
    def draw_grid(self):
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                x1, y1 = i*self.CELL_SIZE, j*self.CELL_SIZE
                x2, y2 = x1+self.CELL_SIZE, y1+self.CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="white", width=2, stipple='gray12', tags="grid")

    #hover effect hovering over the grid with the mouse
    def on_mouse_hover(self, event):
        #deleting prior hovering effets 
        self.canvas.delete("hover")
        #calculating the coordinated of the cell
        x, y = event.x // self.CELL_SIZE, event.y // self.CELL_SIZE
        x1, y1 = x*self.CELL_SIZE, y*self.CELL_SIZE
        x2, y2 = x1+self.CELL_SIZE, y1+self.CELL_SIZE
    
        #creating a new rectangle for the current cell with a border
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=3, tags="hover")

    #right click to add a target
    def on_canvas_click(self, event):
        #calculate which square has been clicked on
        row = event.y // self.CELL_SIZE
        col = event.x // self.CELL_SIZE
        
        #right click for target
        if event.num == 3:
            self.add_target(col, row)

    #left button long press to detect obstacles
    def on_canvas_press(self, event):
        self.press_time = time.time()
        self.press_event = event
        self.press_id = self.master.after(3000, self.on_obstacle_click, event)

    #release to determine the type of click
    def on_canvas_release(self, event):
        elapsed_time = time.time() - self.press_time
        self.master.after_cancel(self.press_id)
        if elapsed_time < 3: #a quick single left click will add a pedestrian
            self.add_pedestrian(event.x // self.CELL_SIZE, event.y // self.CELL_SIZE)

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
