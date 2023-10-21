import random
import sys
import tkinter
from tkinter import Button, Canvas, Entry, IntVar, Label, Menu, Radiobutton
from scenario_elements import Scenario, Pedestrian


class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self):
        self.canvas = None  # Initialize the canvas attribute
        self.random_placement_var = None
        self.x_entry = None
        self.y_entry = None


    # ... (rest of the class)

    def create_scenario(self, ):
        print('create not implemented yet')


    def restart_scenario(self, ):
        print('restart not implemented yet')


    def step_scenario(self, scenario, canvas, canvas_image):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        scenario.update_step()
        scenario.to_image(canvas, canvas_image)

    def add_pedestrian(self,x,y):
        # print('pedestrain not implemented yet')
        """
        Add a pedestrian to the scenario at the specified position (x, y).

        Args:
            x (int): x-coordinate of the pedestrian's position.
            y (int): y-coordinate of the pedestrian's position.
        """
        if 0 <= x < self.scenario.width and 0 <= y < self.scenario.height:
            self.scenario.grid[x, y] = Scenario.NAME2ID['PEDESTRIAN']
            self.scenario.to_image(self.canvas, self.canvas_image)

    def add_target(self, x, y):
        """
        Add a target to the scenario at the specified position (x, y).

        Args:
            x (int): x-coordinate of the target's position.
            y (int): y-coordinate of the target's position.
        """
        if 0 <= x < self.scenario.width and 0 <= y < self.scenario.height:
            self.scenario.grid[x, y] = Scenario.NAME2ID['TARGET']
            self.scenario.to_image(self.canvas, self.canvas_image)

    def add_obstacle(self, x, y):
        """
        Add an obstacle to the scenario at the specified position (x, y).

        Args:
            x (int): x-coordinate of the obstacle's position.
            y (int): y-coordinate of the obstacle's position.
        """
        if 0 <= x < self.scenario.width and 0 <= y < self.scenario.height:
            self.scenario.grid[x, y] = Scenario.NAME2ID['OBSTACLE']
            self.scenario.to_image(self.canvas, self.canvas_image)

    def add_obstacle_handler(self, random_location=True, x=None, y=None):
        random_location = self.random_placement_var.get() == 1  # Check if random placement is selected
        if random_location:
            x = random.randint(0, self.scenario.width - 1)
            y = random.randint(0, self.scenario.height - 1)
        else:
            x = int(self.x_entry.get())  # Use user-specified X coordinate
            y = int(self.y_entry.get())  # Use user-specified Y coordinate
        self.add_obstacle(x, y)

    def add_pedestrian_handler(self, random_location=True, x=None, y=None):
        random_location = self.random_placement_var.get() == 1
        if random_location:
            x = random.randint(0, self.scenario.width - 1)
            y = random.randint(0, self.scenario.height - 1)
        else:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
        self.add_pedestrian(x, y)

    def add_target_handler(self, random_location=True, x=None, y=None):
        random_location = self.random_placement_var.get() == 1
        if random_location:
            x = random.randint(0, self.scenario.width - 1)
            y = random.randint(0, self.scenario.height - 1)
        else:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
        self.add_target(x, y)

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()


    def start_gui(self, ):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry('500x500')  # setting the size of the window
        win.title('Cellular Automata GUI')

        # Create the canvas and assign it to the canvas attribute
        self.canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])
        self.canvas_image = self.canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        self.canvas.pack()


        self.scenario = Scenario(100, 100)  # Create a Scenario object and assign it to self.scenario

        # Create radio buttons for random or specific placement
        self.random_placement_var = IntVar()  # Create the IntVar after initializing the window
        random_radio = Radiobutton(win, text='Random Placement', variable=self.random_placement_var, value=1)
        specific_radio = Radiobutton(win, text='Specific Placement', variable=self.random_placement_var, value=0)
        random_radio.place(x=20, y=50)
        specific_radio.place(x=20, y=75)

        # Create input fields for specific coordinates
        Label(win, text="X Coordinate").place(x=150, y=50)
        self.x_entry = Entry(win)
        self.x_entry.place(x=250, y=50)
        Label(win, text="Y Coordinate").place(x=150, y=75)
        self.y_entry = Entry(win)
        self.y_entry.place(x=250, y=75)



        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        canvas.pack()

        sc = Scenario(100, 100)

        sc.grid[23, 25] = Scenario.NAME2ID['TARGET']
        #sc.grid[23, 45] = Scenario.NAME2ID['TARGET']
        #sc.grid[43, 55] = Scenario.NAME2ID['TARGET']
        sc.recompute_target_distances()

        sc.pedestrians = [
           # Pedestrian((31, 2), 2.3),
           # Pedestrian((1, 10), 2.1),
            Pedestrian((80, 70), 2.1)
        ]

        # can be used to show pedestrians and targets
        sc.to_image(canvas, canvas_image)

        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)

        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario(sc, canvas, canvas_image))
        btn.place(x=20, y=0)
        btn = Button(win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=200, y=0)
        btn = Button(win, text='Create simulation', command=self.create_scenario)
        btn.place(x=380, y=0)

        btn = Button(win, text='Add obstacles', command=self.add_obstacle_handler)
        btn.place(x=380, y=25)
        btn = Button(win, text='Add pedestrians', command=self.add_pedestrian_handler)
        btn.place(x=20, y=25)
        btn = Button(win, text='Add targets', command=self.add_target_handler)
        btn.place(x=200, y=25)

        

        win.mainloop()
