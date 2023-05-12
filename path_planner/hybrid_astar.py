# ---------------------
from scenario.highway import WINDOW_HEIGHT, WINDOW_WIDTH, LANES
import pygame
import numpy as np
import heapq as hq
from scipy import interpolate as itp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import time
from operator import itemgetter
from itertools import repeat

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler('hybrid_astar.log' if __name__ == "__main__" else 'LogFiles/hybrid_astar.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger


class HybridAstar:
    lf = 2  # Front semi-wheelbase
    lr = 2.5  # Rear semi-wheelbase
    length = 15  # length (x-dim) of the car
    width = 8  # width (y-dim) of the car
    velocity = 1  # Directions
    # Possible steering angles (num must be odd to have option of delta=0)
    delta = np.array([np.deg2rad(np.linspace(-45, 45, num=33))]).reshape(-1, 1)
    dt = 4  # Timestep

    def __init__(self):
        # Bounds of the search area
        self.min_y, self.max_y = 0, WINDOW_HEIGHT
        self.reset()

        logger.info('Created {} instance'.format(self.__class__.__name__))

    def reset(self, start_pos: tuple = (0, 0, 0), goal_pos: tuple = (0, 0, 0), vehicle_rects: [pygame.rect] = ()):
        self.open_heap = []  # element of this list is like (cost,node_d)
        self.open_dict = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))
        self.closed_dict = {}  # element of this is like node_d:(cost,node_c,(parent_d,parent_c))
        self.interp_path = ()  # Stores the tck tuple containing the interpolated shortest path.
        self.goal_pos = goal_pos  # Goal position
        self.start_pos = start_pos  # Start position
        self.obstacle_rects = vehicle_rects
        self.obstacles = self.create_obstacles(vehicle_rects) if vehicle_rects else []  # Stores x,y points of obstacles
        self.min_x = start_pos[0]
        self.max_x = goal_pos[0]

    def run(self, start_pos: tuple, goal_pos: tuple, vehicles: [pygame.rect]) -> [np.ndarray, [np.ndarray], int]:
        """Inputs: Start position (x, y, delta), Goal position (x, y, delta), vehicles list(Pygame.rect objects)"""
        # Modify obstacle size to account for Agent dimensions
        vehicle_rects = [car.inflate(self.length, self.width) for car in vehicles if car]
        self.reset(start_pos, goal_pos, vehicle_rects)  # Reset state of all nodes/costs/obstacles

        self.f_current = self.heuristic(np.array(start_pos), np.array(goal_pos))
        hq.heappush(self.open_heap, (self.f_current, start_pos))  # Put the start node/cost into the heapq

        # Add the start node to the node dictionary
        # (total cost, cont. pos., (disc. parent pos., cont. parent pos.), direction, delta)
        self.open_dict[start_pos] = (self.f_current, start_pos, (start_pos, start_pos), 1, 0)

        # Run the algorithm
        timeout = time.time() + 0.07
        while not self.iterate():
            if time.time() > timeout:
                break  # Exit the while loop if taking too long
        else:
            self.success()
            return self.interp_path

    def run2(self, start_pos: tuple, goal_pos: tuple, vehicles: [pygame.rect]) -> [np.ndarray, [np.ndarray], int]:
        """Inputs: Start position (x, y, delta), Goal position (x, y, delta), vehicles list(Pygame.rect objects)"""
        # Modify obstacle size to account for Agent dimensions
        vehicle_rects = [car.inflate(self.length, self.width) for car in vehicles if car]
        self.reset(start_pos, goal_pos, vehicle_rects)  # Reset state of all nodes/costs/obstacles

        self.f_current = self.heuristic(np.array(start_pos[:2]), np.array(goal_pos[:2]))
        hq.heappush(self.open_heap, (self.f_current, start_pos))  # Put the start node/cost into the heapq

        # Add the start node to the node dictionary
        # (total cost, cont. pos., (disc. parent pos., cont. parent pos.), direction, delta)
        self.open_dict[start_pos] = (self.f_current, start_pos, (start_pos, start_pos), 1, 0)

        # Run the algorithm
        timeout = time.time() + 0.07
        while not self.iterate():
            if time.time() > timeout:
                break  # Exit the while loop if taking too long
        else:
            self.success()
            return self.interp_path

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

    def iterate(self):
        def update_heap(valid_nwc2):
            def sort_unique(neighbours):
                """Sorts neighbours from smallest to largest by f_cost.
                Only the first occurrence of each discrete position is kept."""
                sorted = neighbours[np.argsort(neighbours[:, 8])]  # Sort by smallest to largest f_cost
                # Indices of unique elements with smallest f_cost
                _, unique_index = np.unique(sorted[:, :3], return_index=True, axis=0)
                return sorted[unique_index, :]  # Return unique values with the smallest f_cost

            valid_nwc2 = sort_unique(valid_nwc2)  # Sort by smallest to largest f_cost. Keep unique elements.

            open_list = list(self.open_dict.keys())  # Get the list of nodes in the open dictionary
            # Check which neighbours are not in the open dictionary. (Elements are True if not in the open dictionary)
            D = self.compute_boolean_difference(valid_nwc2[:, :3], np.array(open_list)) if open_list else \
                np.ones((np.size(valid_nwc2[:, :3], axis=0), 1), dtype=int)

            # Get current stored cost of items in open dictionary-------------------------
            in_open_dict = valid_nwc2 * np.logical_not(D)
            in_open_dict = valid_nwc2[~np.all(in_open_dict == 0, axis=1)]  # Remove zeros

            previous_f_costs = itemgetter(*tuple(map(tuple, in_open_dict[:, :3])))(self.open_dict) if not D.all() else []
            try:
                # If wanted_keys contains multiple neighbours
                previous_f_costs = np.array(list(map(itemgetter(0), previous_f_costs)))
            except IndexError:
                # Handles "invalid index to scalar variable" (If wanted_keys only contains one neighbour)
                previous_f_costs = previous_f_costs[0]

            # Indices where corresponding f_cost is lower than currently in the dictionary
            lower_cost_indices = np.where(in_open_dict[:, 8] < previous_f_costs)[0]
            lower_cost = in_open_dict[lower_cost_indices]  # Neighbours in the open dictionary with lower cost

            not_in_open_dict = valid_nwc2 * D
            not_in_open_dict = valid_nwc2[~np.all(not_in_open_dict == 0, axis=1)]  # Remove zeros

            update = np.concatenate((not_in_open_dict, lower_cost))  # numpy array of neighbours to update

            # Update open dictionary without for loop--------------
            new_key_values = dict(zip(tuple(map(tuple, update[:, 0:3])),
                                      zip(tuple(update[:, 8]), update[:, [3, 4, 5]],
                                          repeat((self.current_d, self.continuous_pos)), update[:, 7], update[:, 6])))
            self.open_dict.update(new_key_values)

            # Update heapq----- Update contains items: not in open dict, in open dict with lower cost than before.
            update = update[np.argsort(update[:, 8])]
            new_heap = [*zip(update[:, 8], list(map(tuple, update[:, :3])))]

            # Remove nodes from the heap that now have a lower cost
            [self.open_heap.remove(self.open_heap[i]) for discrete in lower_cost[:, :3]
             for i, v in enumerate(self.open_heap) if v[1] == tuple(discrete)]

            self.open_heap.extend(new_heap)
            hq.heapify(self.open_heap)

        def update_heap2(valid_nwc):
            def update_open_dict(neighbours):
                sorted_neighbours = neighbours[np.argsort(neighbours[:, 8])]  # Sort smallest to largest by f_cost
                # Indices of unique elements with smallest f_cost
                _, unique_index = np.unique(sorted_neighbours[:, :3], return_index=True, axis=0)
                valid_ncw_unique = sorted_neighbours[unique_index, :]  # Return unique values with the smallest f_cost

                open_list = list(self.open_dict.keys())  # Get the list of nodes in the open dictionary
                # Check which neighbours are not in the open dictionary
                D = np.full_like(valid_ncw_unique[:, 0], True, dtype=bool) if not open_list else \
                    self.compute_boolean_difference(valid_ncw_unique[:, :3], np.array(open_list)).squeeze()
                not_in_open_dict = valid_ncw_unique[D]
                in_open_dict = valid_ncw_unique[~D]
                previous_f_costs = np.array([self.open_dict[tuple(key)][0] if tuple(key) in self.open_dict else []
                                             for key in in_open_dict[:, :3]])
                lower_cost_indices = np.where(in_open_dict[:, 8] < previous_f_costs)[0]
                lower_cost = in_open_dict[lower_cost_indices]

                # Update open dictionary
                new_key_values = {
                    tuple(row[:3]): (row[8], row[4:7], (self.current_d, self.continuous_pos), row[7], row[6])
                    for row in np.concatenate((not_in_open_dict, lower_cost))}
                self.open_dict.update(new_key_values)

            update_open_dict(valid_nwc)

            # Update heapq----- Update contains items: not in open dict, in open dict with lower cost than before.
            new_heap = [(row[8], tuple(row[:3])) for row in valid_nwc]
            self.open_heap = [v for v in self.open_heap if tuple(v[1]) not in valid_nwc[:, :3]]
            hq.heapify(self.open_heap)
            [hq.heappush(self.open_heap, item) for item in new_heap if item[1] in self.open_dict]

        try:
            self.f_current, self.current_d = hq.heappop(self.open_heap)  # Get total cost and position of current node
        except IndexError:
            logger.warning("Failed to find current node in the heap")
            return

        self.closed_dict[self.current_d] = self.open_dict[self.current_d]  # Add current node to list of explored nodes

        # Check if we found the goal
        finished = np.all(abs(np.array(self.current_d[:2]) - self.goal_pos[:2]) < self.dt) & (
                abs(self.current_d[2] - self.goal_pos[2]) < abs(self.delta[1] - self.delta[0]))

        if not finished:
            self.f_current, self.continuous_pos, _, self.current_velocity, self.current_delta = self.open_dict.pop(
                self.current_d)
            # Get list of possible next positions (Each row corresponds to a neighbour, [dx, dy, dd, cx, cy, cd, d, v])
            neighbours = self.kinematic_model(self.continuous_pos)
            valid_neighbours = self.get_valid_neighbours(neighbours)
            valid_nwc = self.get_node_costs(valid_neighbours, self.obstacles)

            valid_nwc.any() and update_heap(valid_nwc)

        return finished

    def success(self):
        path = []  # Final path
        node = self.current_d
        while tuple(node) != tuple(self.start_pos):
            open_node = self.closed_dict[tuple(node)]  # (total_cost, node_continuous_coords, (parent_discrete_coords, parent_continuous_coords))
            node, parent = open_node[2]
            path.append(parent)
        self.rev_final_path = np.vstack((*reversed(path), self.goal_pos))[::1]
        if len(path) > 3:
            try:
                # noinspection PyTupleAssignmentBalance
                tck, _ = itp.splprep([list(self.rev_final_path[:, i]) for i in range(3)], s=0, k=3)
                self.interp_path = tck
            except TypeError:
                pass

    def success2(self):
        path = []  # Final path
        node = self.current_d
        while tuple(node) != tuple(self.start_pos):
            open_node = self.closed_dict[tuple(node)]  # (total_cost, node_continuous_coords, (parent_discrete_coords, parent_continuous_coords))
            node, parent = open_node[2]
            path.append(parent)
        self.rev_final_path = np.vstack((*reversed(path), self.goal_pos))[::1]
        if len(path) > 3:
            x_path, y_path = self.rev_final_path[:, 0], self.rev_final_path[:, 1]
            try:
                # noinspection PyTupleAssignmentBalance
                tck, _ = itp.splprep([list(x_path), list(y_path)], s=0, k=3)
                self.interp_path = tck
            except TypeError:
                pass

    def get_valid_neighbours(self, neighbours: np.ndarray) -> np.ndarray:
        def check_bounds(neighbours: np.ndarray) -> np.ndarray:
            # Check if neighbours are outside the search area
            bounds = (neighbours[:, 0] > self.min_x) & (neighbours[:, 0] < self.max_x) & \
                     (neighbours[:, 1] > self.min_y) & (neighbours[:, 1] < self.max_y)
            # Convert bounds to a column vector
            return bounds[:, np.newaxis]
        bounds = check_bounds(neighbours)
        collision = self.compute_boolean_difference(neighbours[:, [0, 1]], self.obstacles)
        explored = self.compute_boolean_difference(neighbours[:, :3], np.array([*self.closed_dict]))
        return neighbours * (bounds & collision & explored)

    def get_node_costs(self, valid_neighbours, obstacles):
        def g_local(valid_neighbours, obstacles, current_delta, current_c):
            """
            Calculates the cost of moving from the current node to each neighbour.
            """
            def obstacle_cost(continuous_position: np.ndarray, prox_distance: float) -> np.ndarray:
                difference = continuous_position[:, np.newaxis] - obstacles
                proximity = np.linalg.norm(difference, axis=2)
                obstacle_cost = (prox_distance / proximity) ** 2
                return np.sum(obstacle_cost, axis=1)

            try:
                total_obs_cost = obstacle_cost(valid_neighbours[:, 3:5], 0.7)  # Obstacle cost
            except ValueError:
                total_obs_cost = 0

            steer_cost = np.abs(valid_neighbours[:, 6] - current_delta) / np.max(self.delta) + 1  # Steering angle cost
            reverse_cost = (valid_neighbours[:, 7] < 0) * 100 + 1  # Reverse cost
            # Heading cost
            heading_cost = (np.pi - np.abs((valid_neighbours[:, 5] - current_c[2]) % (2 * np.pi) - np.pi)) / np.pi + 1
            return total_obs_cost * steer_cost * reverse_cost * heading_cost

        # Cost from current to neighbouring nodes
        g_local = g_local(valid_neighbours, obstacles, self.current_delta, self.continuous_pos)
        # Cost from start to the current node
        g_cost = self.f_current - self.heuristic(np.array(self.continuous_pos[:2]), np.array(self.goal_pos[:2]))
        # Calculate the total cost for each neighbour
        f_cost = g_cost + g_local + self.heuristic(valid_neighbours[:, 3:5], np.array(self.goal_pos[:2]))
        # Concatenate the list of costs with the modified list of neighbours
        valid_neighbours_with_cost = np.concatenate((valid_neighbours, f_cost.reshape(-1, 1)), axis=1)
        return valid_neighbours_with_cost[valid_neighbours_with_cost[:, 0] != 0]  # Remove invalid neighbours

    def get_node_costs2(self, valid_neighbours, obstacles):
        def g_cost(valid_neighbours, obstacles):
            """Calculates the cost of moving from the current node to each neighbour."""
            # Obstacle cost------------
            A, B = valid_neighbours[:, 3:5], obstacles
            try: difference = (A[:, np.newaxis] - B).reshape(-1, A.shape[1])
            except ValueError: total_obs_cost = 0
            else:
                sqrdiff = difference ** 2
                sqrdiffsum = np.sum(sqrdiff, axis=1)
                proximity = np.sqrt(sqrdiffsum).reshape(np.size(A, axis=0), -1)

                prox_distance = 3 # A cost of 1 is applied if a neighbour is this many units away from an obstacle
                obstacle_cost = (prox_distance / proximity) ** 2  # Each row corresponds to a neighbour, each column corresponds to a point on an obstacle.
                total_obs_cost = np.sum(obstacle_cost, axis=1)  # The total obstacle cost for each neighbour

            steer_cost = abs(valid_neighbours[:, 6] - self.current_delta) / self.delta.max() + 1  # Steering angle cost. The difference between the steering angle of each neighbour and the current steer angle. Normalised by the max steering angle
            reverse_cost = (valid_neighbours[:, 7] < 0) * 100 + 1 # Penalise reversing
            heading_cost = (abs(np.pi - abs((valid_neighbours[:, 5] - self.continuous_pos[2]) - np.pi)) / np.pi) + 1  # Heading cost

            return total_obs_cost * steer_cost * reverse_cost * heading_cost

        g_local = g_cost(valid_neighbours, obstacles) # Cost from current node to neighbouring nodes
        g_cost = self.f_current - self.heuristic(np.array(self.continuous_pos[:2]), np.array(self.goal_pos[:2]))  # Cost from start to the current node

        g_cost += g_local # Add the local costs to the cost so far
        f_cost = (g_cost + self.heuristic(valid_neighbours[:, 3:5], self.goal_pos[:2])).reshape(-1,1) # Calculate the total cost for each neighbour

        valid_neighbours_with_cost = np.concatenate((valid_neighbours, f_cost),axis=1) # Concatenate the list of costs with the modified list of neighbours.
        return valid_neighbours_with_cost[~np.all(valid_neighbours == 0,axis=1)] # Remove invalid neighbours. Last element in each row of valid2 is the cost associated with that neighbour.

    def kinematic_model(self, current_c):
        beta = np.arctan((self.lf / (self.lf + self.lr)) * np.tan(self.delta)) * self.dt  # Sideslip angle
        # Next x-position (direction * cos(steer_angle + sideslip)) * timestep
        neighbour_x_c = current_c[0] + (self.velocity * np.cos(current_c[2] + beta)) * self.dt
        # Next y-position "" sin(steer_angle + sideslip) ""
        neighbour_y_c = current_c[1] + (self.velocity * np.sin(current_c[2] + beta)) * self.dt
        # New heading angle is the current_heading_angle + (direction * sin(sideslip/b) * timestep)
        neighbour_theta_c = (current_c[2] + (self.velocity * np.sin(beta) / self.lr) * self.dt) % (2 * np.pi)

        neighbours_c = np.column_stack((neighbour_x_c, neighbour_y_c, neighbour_theta_c))  # Cont. pos. of neighbours
        neighbours_d = np.around(neighbours_c, decimals=4).astype(int)  # Discrete position of neighbours
        deltas = np.tile(self.delta, self.velocity)  # Delta values corresponding to each neighbour
        vs = np.tile(self.velocity, self.delta.shape)  # Velocity values corresponding to each neighbour
        return np.column_stack((neighbours_d, neighbours_c, deltas, vs))

    def kinematic_model2(self, current_c):
        def cell(q):
            q1 = np.around(q[0], decimals=4)
            q3 = np.around(q[1], decimals=4)
            q2 = q[2]
            return np.transpose(np.dstack((q1, q3, q2)), [2, 0, 1])

        def do_calc():
            beta = np.arctan((self.lf / (self.lf + self.lr)) * np.tan(self.delta)) * self.dt  # Sideslip angle
            neighbour_x_c = current_c[0] + (self.velocity * np.cos(
                current_c[2] + beta)) * self.dt  # Next x-position (direction * cos(steer_angle + sideslip)) * timestep
            neighbour_y_c = current_c[1] + (self.velocity * np.sin(
                current_c[2] + beta)) * self.dt  # Next y-position "" sin(steer_angle + sideslip) ""
            neighbour_theta_c = (current_c[2] + (self.velocity * np.sin(beta) / self.lr) * self.dt) % (
                    2 * np.pi)  # New heading angle is the current_heading_angle + (direction * sin(sideslip/b) * timestep)

            return np.stack((neighbour_x_c, neighbour_y_c, neighbour_theta_c))

        neighbours_c = do_calc()  # Continuous position of neighbours
        neighbours_d = cell(neighbours_c)  # Discrete position of neighbours
        deltas = self.delta * np.ones_like(self.velocity)  # Delta values corresponding to each neighbour
        vs = self.velocity * np.ones_like(self.delta)  # Velocity values corresponding to each neighbour
        neighbours = np.concatenate((neighbours_d.T.reshape(-1, 3), neighbours_c.T.reshape(-1, 3),
                                     deltas.T.reshape(-1, 1), vs.T.reshape(-1, 1)), axis=1)

        return neighbours

    @staticmethod
    def heuristic(position: np.ndarray, target: np.ndarray) -> float:
        """Heuristic function used by the algorithm. position can be an 2D numpy array."""
        h_cost = np.sum(((position - target) ** 2), axis=1 if position.ndim - 1 else None)
        return np.sqrt(h_cost)

    @staticmethod
    def compute_boolean_difference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Computes the boolean difference between two arrays, A and B.
        Returns a boolean array with shape (n, 1), where n is the number of rows in A.
        Checks if each row of A exists in B, returns 0 if True.
        Output is a logical vector of length 'num rows of A'. A and B must be of size (m,k), (n,k)."""
        try:
            diff = np.logical_not(np.all(A[:, np.newaxis] == B, axis=2)).all(axis=1)
        except ValueError as e:
            diff = np.ones_like(A[:, 0], dtype=bool)
            # raise ValueError("Arrays must have the same number of columns.") from e
        return diff.reshape(-1, 1)

    def plot_solution(self, path, obstacles):
        """Plots the solution in a new figure window.
        Path is a nx3 numpy array of x,y points. Obstacles is a list of pygame rect objects."""
        x_path, y_path = path[:, 0], path[:, 1]

        # Discretise the spline for plotting
        x_new, y_new, delta_new = itp.splev(np.linspace(0, 1, self.goal_pos[0]-self.start_pos[0] + 1), self.interp_path)

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
    start_pos = (0, 102, 0)
    goal_pos = (125, 89, 0)

    path_planner = HybridAstar()  # Initialise Hybrid A* algorithm

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
    path_planner.run(start_pos, goal_pos, vehicle_list)  # Get the shortest path to the goal
    print(time.perf_counter() - start)

    path_planner.plot_solution(path_planner.rev_final_path, vehicle_list)


if __name__ == "__main__":
    main()
