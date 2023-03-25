"""
f(n) = g(n) + h(n)
n is the next node on the path.
g(n) is the cost of the path from the start node to n
h(n) is a heuristic function that estimates the cost of the cheapest path from n to the goal.
Heuristic function should never overestimate the actual cost to get to the goal.

Implementation:
Use a priority queue (open set or fringe) to perform repeated selection of minimum cost nodes to expand.
At each step:
    - Remove the node with the lowest f(n) value.
    - If the node is the goal node, Finish
    - Update the f and g values of its neighbours.
    - Add the neighbours to the queue.
    - Keep track of the current sequence of nodes.

Inside loop:
Inputs: Current position, Goal position, obstacle locations
"""
import time

from config import WINDOW_WIDTH, WINDOW_HEIGHT, LANES

# ----------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy import interpolate as itp
import pygame
from queue import PriorityQueue, Empty
import inspect
from timeit import default_timer as timer

from itertools import count

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler('normal_astar.log' if __name__ == "__main__" else 'LogFiles/normal_astar.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger


class Node:
    """Nodes for A* algorithm."""
    colours = {
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (0, 0, 255),
        "YELLOW": (255, 255, 0),
        "WHITE": (255, 255, 255),
        "BLACK": (0, 0, 0),
        "PURPLE": (128, 0, 128),
        "ORANGE": (255, 165, 0),
        "GREY": (128, 128, 128),
        "SILVER": (192, 192, 192),
        "GAINSBORO": (220, 220, 220),
        "TURQUOISE": (64, 224, 208)
    }

    def __init__(self, row, col, width, height, total_rows, total_cols):
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.x = col * width  # x-coordinate of node converted from row position (increases right)
        self.y = row * height  # y-coordinate of node converted from column position (increases down)
        self.colour = self.colours["WHITE"]
        self.neighbours = []

    def get_pos(self):
        return self.row, self.col  # Determines row and column position of a particular node

    def is_closed(self):
        return self.colour == self.colours["RED"]  # Visited nodes = Red

    def is_open(self):
        return self.colour == self.colours["GREEN"]  # Nodes to be visited = Green

    def is_obstacle(self):
        return self.colour == self.colours["BLACK"]  # Obstacles = Black

    def is_1_from_obstacle(self):
        return self.colour == self.colours["SILVER"]  # Nodes 1 away from obstacle = Silver

    def is_2_from_obstacle(self):
        return self.colour == self.colours["GAINSBORO"]  # Nodes 2 away from obstacle = GAINSBORO

    def is_start(self):
        return self.colour == self.colours["ORANGE"]  # Start node = Orange

    def is_goal(self):
        return self.colour == self.colours["TURQUOISE"]  # Goal node = Turquoise

    def reset(self):
        self.colour = self.colours["WHITE"]  # Colour all reset nodes white

    def make_closed(self):
        self.colour = self.colours["RED"]  # Colour all visited nodes red

    def make_open(self):
        self.colour = self.colours["GREEN"]  # Colour all nodes to be visited green

    def make_obstacle(self):
        self.colour = self.colours["BLACK"]  # Colour all non-traversable nodes (i.e. obstacles) black

    def make_1_from_obstacle(self):
        self.colour = self.colours["SILVER"]  # Colour all nodes 1 away from obstacle silver

    def make_2_from_obstacle(self):
        self.colour = self.colours["GAINSBORO"]  # Colour all nodes 2 away from obstacle gainsboro

    def make_start(self):
        self.colour = self.colours["ORANGE"]  # Colour start node orange

    def make_goal(self):
        self.colour = self.colours["TURQUOISE"]  # Colour goal node turquoise

    def make_path(self):
        self.colour = self.colours["PURPLE"]  # Colour nodes of shortest path purple

    def draw(self, window):
        pygame.draw.rect(window, self.colour, (self.x, self.y, self.width, self.height))

    def update_neighbours(self, grid):
        """Considers movements to any of the 8 surrounding nodes."""
        neighbours = []
        row = self.row
        col = self.col
        total_rows = self.total_rows
        total_cols = self.total_cols
        append = neighbours.append

        # Criteria for moving DOWN 1 row
        # If no obstacle in node below current node and bottom row not yet reached, move down 1 row
        if row < total_rows - 1 and not grid[row + 1][col].is_obstacle(): append(grid[row + 1][col])

        # Criteria for moving UP 1 row
        # If no obstacle in node above current node and top row not yet reached, move up 1 row
        if row > 0 and not grid[row - 1][col].is_obstacle(): append(grid[row - 1][col])

        # Criteria for moving RIGHT 1 column
        # If no obstacle in node to right of current node and rightmost column not yet reached, move right 1 column
        if col < total_cols - 1 and not grid[row][col + 1].is_obstacle(): append(grid[row][col + 1])

        # Criteria for moving LEFT 1 column
        # If no obstacle in node to left of current node and leftmost column not yet reached, move left 1 column
        if col > 0 and not grid[row][col - 1].is_obstacle(): append(grid[row][col - 1])

        # Criteria for moving diagonally
        if row > 0 and not grid[row - 1][col].is_obstacle():
            if (col > 0 and not grid[row][col - 1].is_obstacle()) and not grid[row - 1][col - 1].is_obstacle(): append(grid[row - 1][col - 1]) # UP LEFT
            if (col < total_cols - 1 and not grid[row][col + 1].is_obstacle()) and not grid[row - 1][col + 1].is_obstacle(): append(grid[row - 1][col + 1]) # UP RIGHT

        if row < total_rows - 1 and not grid[row + 1][col].is_obstacle():
            if (col > 0 and not grid[row][col - 1].is_obstacle()) and not grid[row + 1][col - 1].is_obstacle(): append(grid[row + 1][col - 1]) # DOWN LEFT
            if (col < total_cols - 1 and not grid[row][col + 1].is_obstacle()) and not grid[row + 1][col + 1].is_obstacle(): append(grid[row + 1][col + 1]) # DOWN RIGHT

        self.neighbours = neighbours

    def __lt__(self, other):
        return False


class Astar:
    Draw = False  # Draw the nodes on the pygame window

    length = 15  # length (x-dim) of the car
    width = 8  # width (y-dim) of the car

    def __init__(self):
        self.rows, self.cols = 60, 170  # Grid density (number of rows/columns, should be factors of HEIGHT/WIDTH)
        self.width, self.height = WINDOW_WIDTH, WINDOW_HEIGHT  # Width, Height of the pygame window
        self.row_gap = WINDOW_HEIGHT // self.rows  # Distance between rows (pixels)
        self.col_gap = WINDOW_WIDTH // self.cols  # Distance between cols (pixels)
        self.grid = self.make_grid()  # Create the grid of nodes
        self.reset()

        logger.info('Created {} instance'.format(self.__class__.__name__))

    def reset(self):
        grid = self.grid
        [grid[i][j].reset() for i in range(self.rows + 1) for j in range(self.cols + 1)] # Reset the colour of each node

        self.interp_path = () # Stores the tck tuple containing the interpolated shortest path.
        self.path_pos, self.x_path, self.y_path, = [], [], [] # Used in path(path_pos, x_path, y_path), algorithm and main functions

        self.g = {} # Dictionary to store the cost of each node.
        self.f = self.g.copy() # Set the total cost of each node to infinity.

        self.obs_xy = ([], []) # Lists of ([row], [col]) values of obstacles
        self.count = count() # Number of nodes explored
        self.open_set = PriorityQueue() # Queue of nodes to be explored next. Entries are kept sorted by heapq module. Lowest valued entry is returned next.
        self.open_set_hash = {} # Keeps track of nodes in the priority queue (Same information as self.open_set)
        self.came_from = {} # Dictionary of preceding nodes
        self.current = 0 # Stores the current node

    def make_grid(self):
        """Creates a grid of nodes"""
        grid = [[Node(i, j, self.col_gap, self.row_gap, self.rows, self.cols) for j in range(self.cols + 1)] for i in range(self.rows + 1)]
        return grid

    def transform_coord(self, pos):
        """Converts a position in pixels to (row, col)"""
        x, y = pos
        row = y // self.row_gap
        col = x // self.col_gap
        return int(row), int(col)

    def make_node(self, pos):
        """Accepts a raw (x, y) position and returns the corresponding node and its primary and secondary neighbours"""
        row, col = self.transform_coord(pos)
        try: node = self.grid[row][col]  # Finds row and column of node
        except IndexError: pass
        else:
            neighbours, neighbours_2 = [], []
            if row > 0: neighbours.append(self.grid[row - 1][col])
            if row < self.rows - 1: neighbours.append(self.grid[row + 1][col])
            if col > 0: neighbours.append(self.grid[row][col - 1])
            if col < self.cols - 1: neighbours.append(self.grid[row][col + 1])

            if row > 1: neighbours_2.append(self.grid[row - 2][col])
            if row < self.rows - 2: neighbours_2.append(self.grid[row + 2][col])
            if col > 1: neighbours_2.append(self.grid[row][col - 2])
            if col < self.cols - 2: neighbours_2.append(self.grid[row][col + 2])

            if row > 0 and col > 0: neighbours.append(self.grid[row - 1][col - 1])
            if row < self.rows - 1 and col < self.cols - 1: neighbours.append(self.grid[row + 1][col + 1])
            if row > 0 and col < self.cols - 1: neighbours.append(self.grid[row - 1][col + 1])
            if row < self.rows - 1 and col > 0: neighbours.append(self.grid[row + 1][col - 1])

            return node, neighbours, neighbours_2

    def create_obstacles2(self, vehicle_list):
        """Accepts a list of pygame rectangle objects. Each point on the rectangle is converted to an obstacle node.
        points = ['topleft', 'bottomleft', 'topright', 'bottomright', 'midtop', 'midleft', 'midbottom', 'midright', 'center']"""
        x_obs, y_obs = [], []  # Empty lists to store obstacle x,y values
        # points = {23, 24, 25, 31, 32, 33, 34, 38, 39}  # Index of inbuilt points in pygame.rect (python 3.7)
        points = {5, 6, 7, 13, 14, 15, 16, 20, 21}  # Index of inbuilt points in pygame.rect (python 3.11)
        # Get the list of attributes associated with the vehicle rects.
        locations = (inspect.getmembers(vehicle, lambda a: not (inspect.isroutine(a))) for vehicle in vehicle_list)
        # Get all the discrete points of the pygame rects.
        points_list = (veh[point][1] for veh in locations for point in points)

        # points_list = self.create_obstacles(vehicle_list)
        for point in points_list:  # Update neighbours of each obstacle point
            try:
                node, neighbours, neighbours_2 = self.make_node(point)  # Create obstacle node
            except TypeError:
                continue  # Handle TypeError: cannot unpack non-iterable NoneType object
            else:
                if node.get_pos() == self.goal.get_pos():
                    return None
                else:
                    # If immediate neighbour is not an obstacle, make it dark grey
                    [neighbour.make_1_from_obstacle() for neighbour in neighbours if not neighbour.is_obstacle()]
                    # If the secondary neighbour isn't - or isn't next to - an obstacle, make it light grey
                    [neighbour_2.make_2_from_obstacle() for neighbour_2 in neighbours_2 if not neighbour_2.is_obstacle()
                     and not neighbour_2.is_1_from_obstacle()]
                    node.make_obstacle()  # Change node colour to black

                    row, col = self.transform_coord(point)
                    x, y = [col + 1 / 2, ((self.rows - 1) - row) + 1 / 2]  # (x, y) position of the obstacle
                    x_obs.append(x), y_obs.append(y)  # Update the list of obstacle x values

        return x_obs, y_obs

    @staticmethod
    def create_obstacles(car_list: [pygame.rect], area: bool = False) -> np.ndarray:
        """
        Accepts a list of Pygame.rect objects.
        If area: Returns a numpy array of x, y points filling the area with each rectangle
        Else: Returns a numpy array of x, y points on the perimeter of each rectangle.
        """
        if area:
            obs_xy = [np.transpose([np.tile(np.arange(car.left, car.right), car.height),
                                    np.repeat(np.arange(car.top, car.bottom), car.width)]) for car in car_list]
            return np.concatenate(obs_xy, axis=0)
        else:
            obs_points = []
            for car in car_list:
                left, top, width, height = car.left, car.top, car.width, car.height
                obs_points.extend([(left + x, top) for x in range(width)])  # Top side
                obs_points.extend([(left, top + y) for y in range(height)])  # Left side
                obs_points.extend([(left + x, top + height - 1) for x in range(width)])  # Bottom side
                obs_points.extend([(left + width - 1, top + y) for y in range(height)])  # Right side
            return np.array(obs_points)

    def run(self, start_pos: tuple, goal_pos: tuple, vehicles: [pygame.rect]) -> [np.ndarray, [np.ndarray], int]:
        """Runs the A* algorithm. obstacles is a list of pygame rect objects."""
        a = goal_pos  # (x, y)
        b = (self.width, self.height)  # (x, y)
        if all(ai < bi for ai, bi in zip(a, b)):
            # Modify obstacle size to account for Agent dimensions
            vehicle_rects = [car.inflate(self.length, self.width) for car in vehicles if car]

            self.reset()  # Reset the state of all nodes and their corresponding costs.
            self.start_pos, self.goal_pos = start_pos[:2], goal_pos[:2]  # x,y start and goal position.
            try:
                self.start, _, _ = self.make_node(self.start_pos)  # Start node
                self.goal, _, _ = self.make_node(self.goal_pos)  # Goal node
            except TypeError:
                pass
            else:
                self.x_goal = self.goal.get_pos()[1] + 1 / 2
                self.y_goal = self.goal.get_pos()[0] + 1 / 2
                self.obs_xy = self.create_obstacles2(vehicle_rects)  # List of (x, y) values of obstacles
                if self.obs_xy:
                    self.start.make_start(), self.goal.make_goal()  # Change node colours

                    self.g[self.start] = 0 # Set the cost of the start node to 0
                    self.f[self.start] = self.h(self.start.get_pos(), self.goal.get_pos())  # Total cost at start node is just the heuristic cost.

                    self.open_set.put((self.f[self.start], next(self.count), self.start))  # Add the start node to the priority queue. will be explored first (total cost, node_number, node)
                    self.open_set_hash = {self.start}  # Keeps track of nodes in the priority queue (Same information as self.open_set)

                    # __name__ == "__main__" and self.draw and self.draw() # Draws grid on window
                    self.algorithm()  # Run the algorithm
                    return self.interp_path

    def algorithm(self):
        # While the priority queue isn't empty
        while not self.open_set.empty():
            # Get the next node in the priority queue (the node was stored at index 2)
            self.current = self.open_set.get()[2]
            self.current.update_neighbours(self.grid)  # Update the neighbours of the current node
            self.open_set_hash.remove(self.current)  # Remove current node from open set

            # If we haven't found the goal node, iterate through each neighbour of the current node.
            self.success() if self.current == self.goal else self.iterate()
            # If current node is not start node, colour it red
            self.current != self.start and self.current.make_closed()

    def h(self, p1, p2):
        """Heuristic defined as Euclidean distance between goal and current nodes."""
        x1, y1 = p1
        x2, y2 = p2
        # Manhattan distance = Distance in x-direction + Distance in y-direction
        # return abs(x2 - x1) + abs(y2 - y1)

        # Octile heuristic
        x, y = abs(x2 - x1), abs(y2 - y1)*2
        e = 1 / (self.cols ** 5.8)  # Breaks ties between nodes with same f-value by favouring nodes closer to goal.
        hO = (max(x, y) + (np.sqrt(2) - 1) * min(x, y)) * (1 + e)
        # return (max(x, y) + (np.sqrt(2) - 1) * min(x, y)) * (1 + e)

        # Heuristic = Euclidean distance * f-score tie-breaker factor p
        e = 1 / (self.cols ** 2)  # Breaks ties between nodes with same f-value by favouring nodes closer to goal
        hE = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * (1 + e)
        # return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * (1 + e)

        return (hO)

    def iterate(self):
        """Iterates through each neighbour of the current node"""
        # Iterate through each neighbour of the current node
        current = self.current
        for neighbour in current.neighbours:
            # Apply a cost if a neighbouring node is near an obstacle.
            if neighbour.is_1_from_obstacle(): cost = 10
            elif neighbour.is_2_from_obstacle(): cost = 5
            else: cost = 1

            temp_g = self.g[current] + cost # Update the provisional cost of the current node.

            try: self.g[neighbour]
            except KeyError: self.g[neighbour] = float('inf') # If the neighbour hasn't been explored set its cost to inf.

            """If the g cost of the current node is less than the cost of the neighbour, the neighbour comes from the current node.
            The total cost of the neighbour is its g cost plus the h cost."""
            # If the cost of the current node is less than the cost of the neighbouring node.
            if temp_g < self.g[neighbour]:  # Update cost of node so that it is minimum possible value
                self.came_from[neighbour] = current  # Neighbouring node comes from current node
                self.g[neighbour] = temp_g
                self.f[neighbour] = temp_g + self.h(neighbour.get_pos(), self.goal.get_pos())

                """Add the neighbour to the open list if it hasn't been searched."""
                if neighbour not in self.open_set_hash and not neighbour.is_closed():
                    self.open_set.put((self.f[neighbour], next(self.count), neighbour))  # Add the node to the priority queue
                    self.open_set_hash.add(neighbour)
                    neighbour.make_open()  # Change node colour to green

    def success(self):
        while not self.open_set.empty():  # Empty the priority queue.
            try:
                self.open_set.get(False)
            except Empty:
                continue
            self.open_set.task_done()

        # self.path_pos.append((self.x_goal, self.y_goal))  # Add the (x, y) position of the goal to the list of waypoints. Probably inaccurate since determined from low res (row, col) position

        # Get shortest path
        while self.current in self.came_from:  # Keep drawing path until all parent nodes are coloured purple
            self.current = self.came_from[self.current]  # Back-propagate through path nodes in the shortest path.
            y, x = [n + 0.5 for n in self.current.get_pos()]
            # self.current.make_path()  # Sets purple colour
            self.path_pos.append((x, y))  # Add the current node position to the list of path waypoints.

        [(self.x_path.append(insta_pos[0]), self.y_path.append(insta_pos[1])) for insta_pos in self.path_pos[::-1]]

        # Convert back to pygame window coords.
        x_path, y_path = [x * self.col_gap for x in self.x_path], [y * self.row_gap for y in self.y_path]

        # self.start.make_start(), self.goal.make_goal()  # Ensures start/goal nodes original colour when drawing path
        self.rev_final_path = np.vstack((self.start_pos, *list(zip(x_path, y_path)), self.goal_pos))[::1]
        x_path, y_path = self.rev_final_path[:, 0], self.rev_final_path[:, 1]
        try:
            # Draws a cubic spline between the path waypoints. s determines the amount of smoothing (0 is none).
            # tck is a tuple containing (vector of knots, B-line coeffs, degree of the spline).
            # u is the weighted sum of squared residuals of the spline approximation.
            # noinspection PyTupleAssignmentBalance
            tck, _ = itp.splprep([x_path, y_path], s=0, k=3)
            self.interp_path = tck
        except TypeError:
            pass

    def path(self):
        """Sketches the shortest path from goal node to start node."""
        while self.current in self.came_from:  # Keep drawing path until all parent nodes are coloured purple
            self.current = self.came_from[self.current]  # Back-propagate through path nodes in the shortest path.
            y, x = [n + 0.5 for n in self.current.get_pos()]
            # self.current.make_path()  # Sets purple colour
            self.path_pos.append((x, y))  # Add the current node position to the list of path waypoints.

        [(self.x_path.append(insta_pos[0]), self.y_path.append(insta_pos[1])) for insta_pos in self.path_pos[::-1]]

    def plot_solution(self, path, obstacles):
        """Plots the solution in a new figure window.
        Path is a nx3 numpy array of x,y points. Obstacles is a list of pygame rect objects."""
        x_path, y_path = path[:, 0], path[:, 1]

        # Discretise the spline for plotting
        x_new, y_new = itp.splev(np.linspace(0, 1, self.goal_pos[0]-self.start_pos[0] + 1), self.interp_path)

        # Plot the path on a new figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_new, y_new, "-", color="black", label="Smoothed Path", linewidth=0.5)
        ax.plot(x_path, y_path, "|", color="black", label="Waypoints", markersize=4, linewidth=0.1)
        ax.plot(x_path[0], y_path[0], "o", color="k", label="Start", markersize=5, linewidth=0.2)
        ax.plot(x_path[-1], y_path[-1], ">", color="k", label="Goal", markersize=5, linewidth=0.2)

        rectangles = [plt.Rectangle((obs.left, obs.top), obs.width, obs.height) for obs in obstacles]
        patches = PatchCollection(rectangles, fc='k')
        ax.add_collection(patches)
        obs_handle = mpatches.Patch(color='black', label='Obstacles')  # Manually define a new patch

        # Plot lanes:
        ax.set_yticks([*LANES.values()], labels=[*LANES], minor=False, fontsize=20)

        plt.xlabel("Distance", fontsize=20)
        plt.ylabel("Lanes", fontsize=20)

        # Draw legend
        handles, labels = ax.get_legend_handles_labels()  # Get existing legend handles
        handles.append(obs_handle)  # Add the obstacles patch to the list of handles
        plt.legend(handles=handles, loc='best', fancybox=False, shadow=True, fontsize=20)  # Plot the legend

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.25)
        ax.grid(which='major', alpha=0.5)

        plt.xlim([0, 270])
        plt.ylim([0, 120])

        ax.invert_yaxis()

        plt.show()


def main():
    start_pos = (0, 102)  # Start and end position of the algorithm.
    goal_pos = (125, 89)

    path_planner = Astar()  # Initialise the Astar algorithm

    def get_rect_around(centre_point, width, height):
        """ Return a pygame.Rect of size width by height, centred around the given centre_point """
        rectangle = pygame.Rect(0, 0, width, height)  # make new rectangle
        rectangle.center = centre_point  # centre rectangle
        return rectangle

    xs = [100, 250, 250]
    ys = [102, 89, 102]
    ws = [14, 14, 14]
    hs = [6, 5, 6]
    vehicle_list = [get_rect_around((rect[0], rect[1]), rect[2], rect[3]) for rect in zip(xs, ys, ws, hs)]

    start = time.perf_counter()
    path = path_planner.run(start_pos, goal_pos, vehicle_list)  # Run the algorithm
    print(time.perf_counter() - start)

    # sol_path = np.vstack((path[1][0], path[1][1])).T

    path_planner.plot_solution(path_planner.rev_final_path, vehicle_list)


if __name__ == "__main__":
    main()

# import config
# from config import LANES
#
# # -------------
# import pygame
#
# import logging
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
# file_handler = logging.FileHandler('hybrid_astar.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)  # Add file handler to the logger
#
#
# class NormalAstar:
#     def __init__(self):
#         self.min_y, self.max_y = 0, config.WINDOW_HEIGHT
#         self.reset()
#
#         logger.info('Created {} instance'.format(self.__class__.__name__))
#
#     def reset(self, start_pos: tuple = (0, 0, 0), goal_pos: tuple = (0, 0, 0), vehicle_rects: [pygame.rect] = ()):
#         pass
#
#     def run(self, start_pos: tuple, goal_pos: tuple, vehicles: list):
#         """Inputs: Start position (x, y), Goal position (x, y), vehicles list(pygame.rect objects)"""
#         pass
#
#     @staticmethod
#     def create_obstacles():
#         pass
#
#     def iterate(self):
#         pass
#
#     def success(self):
#         pass
#
#     def get_node_costs(self):
#         pass
#
#     @staticmethod
#     def heuristic(position, target) -> float:
#         pass
#
#     def plot_solution(self, path, obstacles):
#         pass
#
#
# def main():
#     global WINDOW_WIDTH, WINDOW_HEIGHT
#     WINDOW_WIDTH, WINDOW_HEIGHT = 1536, 120
#     start_pos = (0, 102, 0)
#     goal_pos = (125, 89, 0)
#
#     path_planner = NormalAstar()  # Initialise Normal A* algorithm
#
#     def getRectAround(centre_point, width, height):
#         """ Return a pygame.Rect of size width by height, centred around the given centre_point """
#         rectangle = pygame.Rect(0, 0, width, height)  # make new rectangle
#         rectangle.center = centre_point  # centre rectangle
#         return rectangle
#
#     xs = [100, 250, 250]
#     ys = [102, 89, 102]
#     ws = [14, 14, 14]
#     hs = [6, 5, 6]
#     vehicle_list = [getRectAround((rect[0], rect[1]), rect[2], rect[3]) for rect in zip(xs, ys, ws, hs)]
#
#     start = time.perf_counter()
#     path_planner.run(start_pos, goal_pos, vehicle_list)  # Get the shortest path to the goal
#     print(time.perf_counter() - start)
#
#     path_planner.plot_solution(path_planner.rev_final_path, vehicle_list)
#
#
# if __name__ == "__main__":
#     main()
