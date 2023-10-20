import sys
import tkinter
from tkinter import Button, Canvas, Menu
from scenario_elements import Scenario, Pedestrian


class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def create_scenario(self, ):
        print('create not implemented yet')


    def restart_scenario(self, ):
        print('restart not implemented yet')

    def load_simulation(self, path):
        self.sc =  Scenario(0, 0, file_path=path)
        self.sc.to_image(self.canvas, self.canvas_image)



    def step_scenario(self):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        self.sc.update_step()
        self.sc.to_image(self.canvas, self.canvas_image)


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
        win.geometry('600x600')  # setting the size of the window
        win.title('Cellular Automata GUI')

        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='New', command=self.create_scenario)
        file_menu.add_command(label='Restart', command=self.restart_scenario)
        file_menu.add_command(label='Close', command=self.exit_gui)

        self.canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        self.canvas_image = self.canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        self.canvas.pack()

        self.sc = Scenario(100, 100)

        self.sc.draw_gridlines(self.canvas)

        self.sc.grid[23, 25] = Scenario.NAME2ID['TARGET']
        self.sc.grid[23, 45] = Scenario.NAME2ID['TARGET']
        self.sc.grid[43, 55] = Scenario.NAME2ID['TARGET']
        self.sc.recompute_target_distances()

        self.sc.pedestrians = [
            Pedestrian((31, 2), 2.3),
            Pedestrian((1, 10), 2.1),
            Pedestrian((80, 70), 2.1)
        ]

        # can be used to show pedestrians and targets
        self.sc.to_image(self.canvas, self.canvas_image)

        # can be used to show the target grid instead
        # sc.target_grid_to_image(canvas, canvas_image)

        btn = Button(win, text='Step simulation', command=lambda: self.step_scenario())
        btn.place(x=20, y=10)
        btn = Button(win, text='Restart simulation', command=self.restart_scenario)
        btn.place(x=160, y=10)
        btn = Button(win, text='Create simulation', command=self.create_scenario)
        btn.place(x=315, y=10)
        btn = Button(win, text='Load simulation', command=lambda: self.load_simulation('scenarios/test_scenario.json'))
        btn.place(x=470, y=10)


        win.mainloop()
