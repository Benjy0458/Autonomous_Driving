# =================================
# Imports
# =================================
import os
from math import pi, sin, cos, sqrt, atan2
from itertools import product
import matplotlib.pyplot as plt
import pygraphviz as pgv
from transpose_dict import TD
import numpy as np
import pygame
import time
import random
import curses
import logging
from multiprocessing import Process, Queue
import ctypes

import scenario.roundabout as scene
from navigation import shortest_path
from controllers import IDM, OptimalControl
from utils import RepeatedTimer
import plotting

ctypes.windll.user32.SetProcessDPIAware()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler(f'LogFiles/Roundabout/{__name__}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger

# =================================
# Global variables
# =================================

# =================================
# Classes
# =================================


class World:
    """A class that handles the pygame simulation environment"""
    def __init__(self):
        self.display_surface = World.init_pygame(caption='Autonomous Roundabout Driving')  # Open a named Pygame window
        self.raw_image = pygame.image.load(rf'{scene.BACKGROUND_IMAGE}').convert()
        self.display_info = pygame.display.Info
        scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT = self.display_info().current_w, self.display_info().current_h
        self.image = pygame.transform.scale(self.raw_image, (scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT))  # Scale the background image

        self.display_surface.blit(self.image, (0, 0))  # Fill the pygame window with background image
        self.road_network = None
        self.TM = None  # Create a new instance of the traffic manager
        self.start_time = 0  # Stores the start time of the simulation
        logger.info(f'Created new instance of the simulation environment.')

    def game_loop(self, live_data: bool = False) -> None:
        self.start_time = time.time()  # Store the start time of the simulation
        clock = pygame.time.Clock()  # Controls the frame rate

        if live_data:
            queue = Queue()  # Stores data for the live graph
            close = Queue()  # Send data to exit the live graph
            live_plot = Process(target=plotting.live_graph, args=(queue, close))
            live_plot.start()

        self.TM and self.TM.timed_spawn.start()  # Spawn vehicles at regular intervals

        running = True
        try:
            logger.info(f'Simulation started.')
            console = curses.initscr()  # Initialise the console object
            prev_w, prev_h = None, None
            pygame.event.set_allowed(pygame.QUIT)  # Set which events are monitored by Pygame
            while running:
                for event in pygame.event.get():  # Decide whether to terminate the simulation
                    if event.type == pygame.QUIT:
                        running = False

                elapsed_time = time.time() - self.start_time  # Calculate elapsed time

                # Get current display dims
                scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT = self.display_info().current_w, self.display_info().current_h
                if prev_w != scene.WINDOW_WIDTH or prev_h != scene.WINDOW_HEIGHT:
                    # todo Update geometry is causing issues with the ellipses
                    self.update_geometry()
                    prev_w, prev_h = scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT

                self.update_position()  # Update the position of vehicles in the window
                self.draw_window(road_network=True)  # Draw features on the pygame window

                pygame.display.update()  # Update the display
                self.caption(elapsed_time, clock)  # Update the window caption
                self.console_output(console)  # Output data to the console

                if live_data and self.TM.vehicles:
                    v = self.TM.vehicles[0]
                    queue.put((elapsed_time, v.xte, v.phi, v.delta, v.velocity, v.t))

                clock.tick_busy_loop(scene.FPS)  # Ensure program maintains desired frame rate
        finally:
            if self.TM:
                self.TM.timed_spawn.stop()  # Stop the spawn vehicles thread
                logger.debug('Successfully terminated the spawn vehicles thread.')

            if live_data:
                close.put(1)
                live_plot.join(timeout=5)
                live_plot.kill()
                logger.debug("Succesfully closed the live plot.")

    def update_geometry(self):
        # Scale the background image
        self.image = pygame.transform.scale(self.raw_image, (scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT))
        if self.TM:
            self.TM.road_network.update_geometry()

    def update_position(self):
        for v in self.TM.vehicles:
            v.update_position()  # Update the position of all vehicles
            if v.r >= v.route_length:
                self.TM.vehicles.remove(v)  # Remove vehicles that have reached the end of their route

    def draw_window(self, background=True, road_network=True):
        background and self.display_surface.blit(self.image, (0, 0))  # Fill the pygame window with background image
        if self.TM and road_network:
            self.TM.road_network.draw(self.display_surface)

        [v.draw(self.display_surface) for v in self.TM.vehicles]  # Draw vehicles on the window

    def caption(self, elapsed_time: float, clock: pygame.time.Clock) -> None:
        cap = []
        sim_cap = f"Elapsed time: {round(elapsed_time)}, FPS: {round(clock.get_fps())}"
        cap.append(sim_cap)
        if self.TM:
            tm_cap = f"Num Vehicles: {len(self.TM.vehicles)}"
            cap.append(tm_cap)

        caption = ", ".join(cap)
        pygame.display.set_caption(caption)

    def console_output(self, console):
        # Console output
        console.clear()
        # Display velocity of the vehicles on each road segment
        segment_occupiers = (f"{i} {[int(o.velocity) for o in segment.occupiers]}\n" for i, segment in
                             enumerate(self.TM.road_network.segments) if segment.occupiers)
        [console.addstr(occupier) for j, occupier in enumerate(segment_occupiers) if j < curses.LINES - 1]
        console.refresh()

    @staticmethod
    def init_pygame(caption: str = "Pygame window") -> pygame.Surface:
        """Returns a new named pygame window."""
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)  # Set the position of the window on the screen
        pygame.init()
        info_object = pygame.display.Info()
        ds = pygame.display.set_mode((scene.WINDOW_WIDTH if scene.WINDOW_WIDTH else info_object.current_w,
                                      scene.WINDOW_HEIGHT if scene.WINDOW_HEIGHT else info_object.current_h),
                                     pygame.RESIZABLE)
        pygame.display.set_caption(caption)
        return ds


class TrafficManager:
    """A class that handles the behaviour of NPC vehicles in the simulation"""

    def __init__(self, road_network):
        self.vehicles = []  # Keeps track of the vehicles in the simulation
        TRAFFIC_DENSITY, TIME_GAIN = 50, 1
        self.timed_spawn = RepeatedTimer(1 / (TRAFFIC_DENSITY * TIME_GAIN), self.spawn_vehicles)
        logger.info('Created new instance of the traffic manager.')

        self.road_network = road_network

    def spawn_vehicles(self):
        """Creates a new vehicle and appends it to the list of vehicles.
        Spawn point can't be the same if spawning multiple vehicles.
        """
        if len(self.vehicles) > 0:
            return

        max_vehicles = 1
        new_vehicles = []  # Temporary list to store our new vehicles.
        number_of_vehicles = random.randint(0, max_vehicles)  # Assign a random number of vehicles to spawn in a range.
        vehicle_categories = random.choices(population=["car", "truck"], cum_weights=[0.7, 1.0], k=number_of_vehicles)

        for v_cat in vehicle_categories:
            route = [self.road_network.segments[int(i)] for i in random.choice(self.road_network.routes)]
            # route = [self.road_network.segments[int(i)] for i in self.road_network.routes[10]]
            new_vehicle = Vehicle(width=10, length=20, route=route)  # Create a new vehicle
            for vehicle in new_vehicles:
                if new_vehicle.route[0] == vehicle.route[0]:
                    # Prevent multiple spawns on the same road segment
                    break
            else:
                # The vehicle object of the car in front
                for route in new_vehicle.route[new_vehicle.active_segment:]:
                    if route.occupiers:  # If a vehicle exists on the road segment
                        new_vehicle.front_car = route.occupiers[-1]  # The last vehicle on the road segment
                        break

                if new_vehicle.front_car:
                    # Check if the new_vehicle is colliding with the front vehicle or agent
                    # collide = new_vehicle.rect.collidelist([new_vehicle.front_car.rect, world.agent.rect]) \
                    #     if world.agent else new_vehicle.rect.collidelist([new_vehicle.front_car.rect])
                    collide = new_vehicle.rect.collidelist([new_vehicle.front_car.rect])
                    if collide != -1:
                        break

                # Add the vehicle to the current segment
                new_vehicle.route[new_vehicle.active_segment].occupiers.append(new_vehicle)
                self.vehicles.append(new_vehicle)


class Vehicle(pygame.sprite.Sprite):
    """Car class for NPC vehicles."""

    def __init__(self, width, length, route):
        pygame.sprite.Sprite.__init__(self)  # Call the parent class constructor

        # Set the driving style and corresponding colour
        self.driving_style = random.choices(population=["safe", "aggressive"], weights=[0.2, 0.8])[0]
        self.colour = pygame.Color(0, 0, 255) if self.driving_style == "aggressive" else pygame.Color(0, 255, 0)

        # Create an image of the vehicle, and fill it with colour
        self.image_orig = pygame.Surface([length, width])
        self.image_orig.set_colorkey([0, 0, 0])
        self.image_orig.fill(self.colour)
        self.image = self.image_orig.copy()
        self.image.set_colorkey([0, 0, 0])

        self.width = width
        self.length = length
        self.route = route
        self.route_length = sum([route.length for route in self.route])  # The total length of the route

        self.max_velocity = scene.SPEED_LIMIT
        self.velocity = scene.SPEED_LIMIT  # Velocity

        self.active_segment = 0  # The index of the current segment in self.routes
        self.segments_completed = 0

        self.front_car = None

        # Spawn location
        self.r = 0  # The distance travelled from the start position
        self.dr = 0  # The distance travelled since the previous timestep
        self.t = self.route[self.active_segment].compute_position(self.r)
        self.x_pos, self.y_pos = self.route[self.active_segment].get_position(self.t)
        dx, dy = self.route[self.active_segment].get_derivative(self.t)
        self.theta = atan2(dy, dx)  # The direction of the vehicle (arctan(gradient))
        self.delta = 0  # The steering angle of the vehicle

        self.rect = self.image.get_rect()  # Fetch the rectangle object with dimensions of the image
        self.rect.center = (self.x_pos, self.y_pos)

        # Initialise speed controller
        self.speed_controller = IDM(driving_style="safe", desired_velocity=self.max_velocity)
        self.lateral_controller = OptimalControl(self.length)

        self.acc = 0  # Initialise acceleration
        self.xte = 0  # Initialise crosstrack error
        self.phi = 0  # Initialise heading error

        # Get the angle of the trajectory
        dx, dy = self.route[self.active_segment].get_derivative(self.t)  # Gradient at the current position
        new_angle = atan2(dy, dx)
        self.traj_angle = new_angle % (2 * pi)

    def update_position(self):
        """Updates the internal state of the vehicle."""

        def get_front_car_data():
            """Returns the relative distance and velocity of the next vehicle the current route."""
            if self.front_car and self.front_car.active_segment < len(self.front_car.route):
                """
                The distance of the front car along its current segment is front_car.r - front_car.segments_completed
                If the front car is not in our segment, need to add the distance to the end of our current segment + the 
                length of any intermediate segments
                The distance to end of our segment is travel_distance - self.r
                """
                # Subtract our distance travelled in the current segment
                front_car_distance = -self.route[self.active_segment].get_length(self.route[self.active_segment].t_min, self.t)

                for segment in self.route[self.active_segment:]:
                    if segment == self.front_car.route[self.front_car.active_segment]:
                        front_car_distance += (self.front_car.r - self.front_car.segments_completed)
                        break
                    else:
                        front_car_distance += segment.length
                front_car_velocity = self.front_car.velocity
            else:
                self.front_car = None
                front_car_distance = 1000
                front_car_velocity = scene.SPEED_LIMIT

            return front_car_distance, front_car_velocity

        def update_vehicle_kinematics():
            """Updates the position, velocity and acceleration of the vehicle.
            Returns the distance travelled since the previous timestep."""
            front_car_dist, front_car_vel = get_front_car_data()
            # Update acceleration in m/s^2
            self.acc = self.speed_controller.speed_control.send([self.velocity, front_car_dist, front_car_vel])

            # Bound the acceleration
            if self.acc > 5:
                self.acc = 5
            elif self.acc < -9:
                self.acc = -9

            self.acc *= 2.237  # Convert to mph/s
            self.velocity += self.acc * (1 / scene.FPS)  # Update velocity
            # Ensure velocity is +ve
            if self.velocity < 0:
                self.velocity = 0
                self.acc = 0

            # Update the position vector
            self.dr = ((self.velocity * 1 / scene.FPS) + 0.5 * self.acc * (1 / scene.FPS) ** 2)
            self.r += self.dr

            # Update the current position in pygame (x, y) coordinates
            self.x_pos += self.dr * cos(self.theta)
            self.y_pos -= self.dr * sin(self.theta)
            return

        def crosstrack_error():
            def bisection(foo: callable, lower: float, upper: float, tol: float = 0.001, max_iter: int = 60) -> float:
                """Implements the bisection method on the provided function, foo."""
                fa = foo(lower)
                F = fa
                n = 0
                while abs(F) > tol and n < max_iter:
                    n += 1
                    t = (upper + lower) / 2
                    F = foo(t)
                    if np.sign(F) == np.sign(fa):
                        lower = t
                    else:
                        upper = t

                return t

            def func(t: float) -> float:
                x, y = self.route[self.active_segment].get_position(t)
                dx, dy = self.route[self.active_segment].get_derivative(t)  # Gradient at the current position
                return (self.y_pos - y) * dy + (self.x_pos - x) * dx

            # The t-value representing the distance traversed along the current road segment
            self.t = bisection(func, self.route[self.active_segment].t_min, self.route[self.active_segment].t_max)
            X_xte, Y_xte = self.route[self.active_segment].get_position(self.t)
            xte = sqrt((self.x_pos - X_xte) ** 2 + (self.y_pos - Y_xte) ** 2)  # The magnitude of the cross-track error

            # Determine the offset direction
            xte_vec = np.array([(X_xte - self.x_pos), (Y_xte - self.y_pos)])
            heading = np.array([self.dr * cos(self.theta), - self.dr * sin(self.theta)])
            Z = -np.cross(heading, xte_vec)
            self.xte = np.sign(Z) * abs(xte)
            return

        def heading_error():
            dx, dy = self.route[self.active_segment].get_derivative(self.t)  # Gradient at the current position
            self.traj_angle = atan2(dy, dx)  # Angle of the trajectory at the current position
            self.phi = self.theta - self.traj_angle  # Heading error
            if self.phi > pi:
                self.phi = self.phi - 2*pi

        def calculate_steering_angle():
            # Calculate gain matrix using optimal control
            k, _, _ = self.lateral_controller.trajectory_control.send(self.velocity)
            delta = -k * [self.xte, self.phi] * [1, 1]  # Controller action is u = -Kx
            self.delta = np.sum(delta)
            max_delta = 24 * 2 * pi / 360
            if self.delta > max_delta:
                self.delta = max_delta
            elif self.delta < -max_delta:
                self.delta = -max_delta

            return

        def update_heading_angle():
            """Updates the heading angle of the vehicle."""
            # Yaw rate = Velocity / Radius
            # Steering angle = length / Radius
            # => Radius = length / steering angle
            try:
                radius = self.length / self.delta
                yaw_rate = (self.velocity / 2.237) / radius
            except ZeroDivisionError:
                yaw_rate = 0
            finally:
                self.theta += yaw_rate * 1 / scene.FPS
                self.theta %= (2 * pi)

            return

        def update_vehicle_image():
            # Adjust the vehicle colour depending on current velocity
            self.colour = pygame.Color(int((1 - self.velocity / 80) * 255), 0, int(
                (self.velocity / 80) * 255)) if self.driving_style == "aggressive" else pygame.Color(
                int((1 - self.velocity / 80) * 255), int((self.velocity / 80) * 255), 0)

            self.image_orig.fill(self.colour)

            # Have to rotate the original image to preserve size of the object
            try:
                self.image = pygame.transform.rotate(self.image_orig, self.theta * 360 / (2 * pi))
            except pygame.error as e:
                logger.warning(f'''Encountered "{e}" error attempting to rotate vehicle image.''')

            self.rect = self.image.get_rect()  # Fetch the rectangle object with dimensions of the image
            self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the vehicle rectangle

        update_vehicle_kinematics()
        crosstrack_error()
        heading_error()
        calculate_steering_angle()
        update_heading_angle()
        update_vehicle_image()

        # todo t must vary between 0-1 for every road segment, with t_start = 0, t_end = 1
        # todo This condition is being satisfied multiple times when reach ellipse segment. (Inverse convert_t method?)
        if self.t == self.route[self.active_segment].t_max:
            try:
                self.route[self.active_segment].occupiers.remove(self)  # Remove the vehicle from the completed segment
            except ValueError:
                logger.warning("Tried to remove a vehicle from the segment list more than once.")

            if self.active_segment < len(self.route) - 1:
                self.active_segment += 1
                # Get the vehicle currently in front, only check vehicles ahead of the current position
                for segment in self.route[self.active_segment:]:
                    if segment.occupiers:  # If a vehicle exists on the segment
                        self.front_car = segment.occupiers[-1]  # The vehicle object of the car in front
                        break
                else:
                    self.front_car = None

                self.route[self.active_segment].occupiers.append(self)  # Add the vehicle to its new segment

    def update_position3(self):
        """
        We have reached the end of the current segment if we have passed the line normal to the end point of the segment.
        The gradient of the normal is -dx/dy, with t=1.
        End position is (x, y), with t=1.

        The
        therefore the line eqn is
        """
        self.colour = pygame.Color(int((1 - self.velocity / 80) * 255), 0, int((self.velocity / 80) * 255)) if self.driving_style == "aggressive" else pygame.Color(int((1 - self.velocity / 80) * 255), int((self.velocity / 80) * 255), 0)
        self.image_orig.fill(self.colour)
        if self.active_segment < len(self.route):
            # Calculate required acceleration from IDM controller
            # If the front car hasn't finished its route
            if self.front_car and self.front_car.active_segment < len(self.front_car.route):
                """
                The distance of the front car along its current segment is front_car.r - front_car.segments_completed
                If the front car is not in our segment, need to add the distance to the end of our current segment + the 
                length of any intermediate segments
                The distance to end of our segment is travel_distance - self.r
                """
                front_car_distance = -(self.r - self.segments_completed)  # Subtract our distance travelled in the current segment
                for segment in self.route[self.active_segment:]:
                    if segment == self.front_car.route[self.front_car.active_segment]:
                        front_car_distance += (self.front_car.r - self.front_car.segments_completed)
                        break
                    else:
                        front_car_distance += segment.length

                # Update acceleration in mph/s
                self.acc = self.speed_controller.speed_control.send([self.velocity, front_car_distance, self.front_car.velocity]) * 2.237
            else:
                self.front_car = None
                front_car_distance = None
                self.acc = self.speed_controller.speed_control.send([self.velocity, 1000, scene.SPEED_LIMIT]) * 2.237  # Update acceleration in mph/s

            # Bound the self.acceleration
            if self.acc / 2.237 > 5:
                self.acc = 5 * 2.237
            elif self.acc / 2.237 < -9:
                self.acc = -9 * 2.237

            self.velocity += self.acc * (1 / scene.FPS)
            # Ensure velocity is +ve
            if self.velocity < 0:
                self.velocity = 0
                self.acc = 0

            dr = ((self.velocity * 1 / scene.FPS) + 0.5 * self.acc * (1 / scene.FPS) ** 2)
            self.r += dr

            self.segments_completed = sum([self.route[i].length for i in range(self.active_segment)]) if self.active_segment else 0  # The total length of fully traversed segments
            travel_distance = self.segments_completed + self.route[self.active_segment].length  # The distance r to the next waypoint

            if self.r >= travel_distance:
                self.route[self.active_segment].occupiers.remove(self)  # Remove the vehicle from the completed segment
                self.active_segment += 1
                if self.active_segment < len(self.route):
                    # Get the vehicle currently in front, only check vehicles ahead of the current position
                    # todo If the front vehicle is not in our segment, another vehicle might cut in front
                    for segment in self.route[self.active_segment:]:
                        if segment.occupiers:  # If a vehicle exists on the segment
                            self.front_car = segment.occupiers[-1]  # The vehicle object of the car in front
                            break
                    else:
                        self.front_car = None

                    self.route[self.active_segment].occupiers.append(self)  # Add the vehicle to the new segment

            if self.active_segment < len(self.route):  # If we haven't reached the goal position
                '''
                Controller action is u = -KX
                X is 
                e: the cross track error
                phi: Yaw (heading) angle relative to the trajectory

                u = -(K1 * e + K2 * phi)
                '''
                self.t = self.route[self.active_segment].compute_position(self.r - self.segments_completed)  # Get current t value

                self.x_pos += dr * cos(self.theta)
                self.y_pos -= dr * sin(self.theta)
                # Cross-track error is the minimum distance between the current position and the trajectory
                # self.xte = 1000
                # for t in np.linspace(0, 1, 1000):
                #     traj_x, traj_y = self.route[self.active_segment].get_position(t)  # (x, y) position
                #     x = self.x_pos - traj_x
                #     y = self.y_pos - traj_y
                #     xte_new = sqrt(x ** 2 + y ** 2)  # Crosstrack error
                #     if abs(self.xte) > abs(xte_new):
                #         try:
                #             grad = y/x
                #         except ZeroDivisionError:
                #             pass
                #         else:
                #             self.xte = xte_new * -grad / abs(grad)  # Update the crosstrack error

                dx, dy = self.route[self.active_segment].get_derivative(self.t)  # Gradient at the current position
                new_angle = atan2(dy, dx)

                self.traj_angle = new_angle % (2 * pi)
                # self.theta = self.traj_angle

                # self.phi = (self.theta - self.traj_angle) % pi
                self.phi = (self.traj_angle - self.theta)
                while abs(self.phi) > pi/2:
                    if self.phi > pi/2:
                        self.phi -= pi/2
                    else:
                        self.phi += pi/2

                def crosstrack_error() -> [float]:
                    def func(t):
                        x, y = self.route[self.active_segment].get_position(t)
                        dx, dy = self.route[self.active_segment].get_derivative(t)  # Gradient at the current position
                        return y1 * dy - y * dy + x1 * dx - x * dx

                    x1, y1 = self.x_pos, self.y_pos
                    tol = 0.001
                    lower, upper = self.route[self.active_segment].t_min, self.route[self.active_segment].t_max
                    fa = func(lower)
                    F = fa
                    n =0
                    while abs(F) > tol and n < 60:
                        n += 1
                        t = (upper + lower) / 2
                        F = func(t)
                        if np.sign(F) == np.sign(fa):
                            lower = t
                        else:
                            upper = t

                    x, y = self.route[self.active_segment].get_position(t)
                    return x, y, sqrt((x1 - x)**2 + (y1 - y)**2)

                X_xte, Y_xte, xte = crosstrack_error()
                """
                XTE_vector
                X_xte = x
                 Y_xte = y
                xte_vec = [(X_xte - self.x_pos), (Y_xte, self.y_pos)]
                heading = [dr * cos(self.theta), - dr * sin(self.theta)]
                Z = cross(heading, xte_vec)
                self.xte = sign(Z) * |xte|
                """

                xte_vec = np.array([(X_xte - self.x_pos), (Y_xte - self.y_pos)])
                heading = np.array([dr * cos(self.theta), - dr * sin(self.theta)])
                Z = -np.cross(heading, xte_vec)
                self.xte = np.sign(Z) * abs(xte)


                # traj_x, traj_y = self.route[self.active_segment].get_position(self.t)  # (x, y) position
                # try:
                #     grad = (self.y_pos - traj_y) / (self.x_pos - traj_x)
                #     self.xte *= -grad / abs(grad)
                # except ZeroDivisionError:
                #     self.xte = 0

                K, P, E = self.lateral_controller.trajectory_control.send(self.velocity)
                delta = -K * [self.xte, self.phi] * [1, 1]
                self.delta = np.sum(delta)
                max_delta = 45
                if self.delta > max_delta:
                    self.delta = max_delta
                elif self.delta < -max_delta:
                    self.delta = -max_delta

                self.delta *= 2 * pi / 360

                # --------------------------------------------
                # Yaw rate = Velocity / Radius
                # Steering angle = length / Radius
                # => Radius = length / steering angle
                try:
                    radius = self.length / self.delta
                    yaw_rate = (self.velocity) / radius
                except ZeroDivisionError:
                    yaw_rate = 0
                finally:
                    self.theta += yaw_rate * 1 / scene.FPS
                    self.theta %= (2*pi)
                # ----------------------------------------------

                # Have to rotate the original image to preserve size of the object
                try:
                    self.image = pygame.transform.rotate(self.image_orig, self.theta * 360 / (2 * pi))
                except pygame.error as e:
                    logger.warning(f'''Encountered "{e}" error attempting to rotate vehicle image.''')
                self.rect = self.image.get_rect()  # Fetch the rectangle object with dimensions of the image
                self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the vehicle rectangle

    def update_position2(self):
        self.colour = pygame.Color(int((1 - self.velocity / 80) * 255), 0, int((self.velocity / 80) * 255)) if self.driving_style == "aggressive" else pygame.Color(int((1 - self.velocity / 80) * 255), int((self.velocity / 80) * 255), 0)
        self.image_orig.fill(self.colour)
        if self.active_segment < len(self.route):
            # Calculate required acceleration from IDM controller
            # If the front car hasn't finished its route
            if self.front_car and self.front_car.active_segment < len(self.front_car.route):
                """
                The distance of the front car along its current segment is front_car.r - front_car.segments_completed
                If the front car is not in our segment, need to add the distance to the end of our current segment + the 
                length of any intermediate segments
                The distance to end of our segment is travel_distance - self.r
                """
                front_car_distance = -(self.r - self.segments_completed)  # Subtract our distance travelled in the current segment
                for segment in self.route[self.active_segment:]:
                    if segment == self.front_car.route[self.front_car.active_segment]:
                        front_car_distance += (self.front_car.r - self.front_car.segments_completed)
                        break
                    else:
                        front_car_distance += segment.length

                # Update acceleration in mph/s
                self.acc = self.speed_controller.speed_control.send([self.velocity, front_car_distance, self.front_car.velocity]) * 2.237
            else:
                self.front_car = None
                front_car_distance = None
                self.acc = self.speed_controller.speed_control.send([self.velocity, 1000, scene.SPEED_LIMIT]) * 2.237  # Update acceleration in mph/s

            # Bound the self.acceleration
            if self.acc / 2.237 > 5:
                self.acc = 5 * 2.237
            elif self.acc / 2.237 < -9:
                self.acc = -9 * 2.237

            self.velocity += self.acc * (1 / scene.FPS)
            # Ensure velocity is +ve
            if self.velocity < 0:
                self.velocity = 0
                self.acc = 0

            dr = ((self.velocity * 1 / scene.FPS) + 0.5 * self.acc * (1 / scene.FPS) ** 2)
            self.r += dr

            self.segments_completed = sum([self.route[i].length for i in range(self.active_segment)]) if self.active_segment else 0  # The total length of fully traversed segments
            travel_distance = self.segments_completed + self.route[self.active_segment].length  # The distance r to the next waypoint

            if self.r >= travel_distance:
                self.route[self.active_segment].occupiers.remove(self)  # Remove the vehicle from the completed segment
                self.active_segment += 1
                if self.active_segment < len(self.route):
                    # Get the vehicle currently in front, only check vehicles ahead of the current position
                    # todo If the front vehicle is not in our segment, another vehicle might cut in front
                    for segment in self.route[self.active_segment:]:
                        if segment.occupiers:  # If a vehicle exists on the segment
                            self.front_car = segment.occupiers[-1]  # The vehicle object of the car in front
                            break
                    else:
                        self.front_car = None

                    self.route[self.active_segment].occupiers.append(self)  # Add the vehicle to the new segment

            if self.active_segment < len(self.route):  # If we haven't reached the goal position
                '''
                Controller action is u = -KX
                X is 
                e: the cross track error
                phi: Yaw (heading) angle relative to the trajectory

                u = -(K1 * e + K2 * phi)
                '''
                self.t = self.route[self.active_segment].compute_position(self.r - self.segments_completed)  # Get current t value

                self.x_pos += dr * cos(self.theta)
                self.y_pos -= dr * sin(self.theta)
                # Cross-track error is the minimum distance between the current position and the trajectory
                # self.xte = 1000
                # for t in np.linspace(0, 1, 1000):
                #     traj_x, traj_y = self.route[self.active_segment].get_position(t)  # (x, y) position
                #     x = self.x_pos - traj_x
                #     y = self.y_pos - traj_y
                #     xte_new = sqrt(x ** 2 + y ** 2)  # Crosstrack error
                #     if abs(self.xte) > abs(xte_new):
                #         try:
                #             grad = y/x
                #         except ZeroDivisionError:
                #             pass
                #         else:
                #             self.xte = xte_new * -grad / abs(grad)  # Update the crosstrack error

                dx, dy = self.route[self.active_segment].get_derivative(self.t)  # Gradient at the current position
                new_angle = -atan2(dy, dx)

                traj_angle = new_angle % (2 * pi)
                self.theta = traj_angle

                self.phi = self.theta - traj_angle

                traj_x, traj_y = self.route[self.active_segment].get_position(self.t)  # (x, y) position
                self.xte = sqrt((self.x_pos - traj_x)**2 + (self.y_pos - traj_y)**2)

                # K, P, E = self.lateral_controller.trajectory_control.send(self.velocity)
                # delta = -K * [self.xte, self.phi] * [1, 1]
                # self.delta = np.sum(delta)
                max_delta = 45
                if self.delta > max_delta:
                    self.delta = max_delta
                elif self.delta < -max_delta:
                    self.delta = -max_delta

                self.delta *= 2 * pi / 360

                # --------------------------------------------
                # Yaw rate = Velocity / Radius
                # Steering angle = length / Radius
                # => Radius = length / steering angle
                # try:
                #     radius = self.length / self.delta
                #     yaw_rate = self.velocity / radius
                # except ZeroDivisionError:
                #     yaw_rate = 0
                # finally:
                #     self.theta += yaw_rate * 1 / scene.FPS
                #     self.theta %= (2*pi)
                # ----------------------------------------------

                # Have to rotate the original image to preserve size of the object
                try:
                    self.image = pygame.transform.rotate(self.image_orig, self.theta * 360 / (2 * pi))
                except pygame.error as e:
                    logger.warning(f'''Encountered "{e}" error attempting to rotate vehicle image.''')
                self.rect = self.image.get_rect()  # Fetch the rectangle object with dimensions of the image
                self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the vehicle rectangle

    def update_geometry(self):
        pass

    def draw(self, display_surface):
        """Draws the vehicle on the window."""
        display_surface.blit(self.image, self.rect)


class RoadNetwork:
    """A class containing road data for the scenario."""
    def __init__(self, graph: dict[dict] = None, spawn_points: [int] = None, goal_points: [int] = None, routes: [tuple] = None, road_data: callable = None):
        """
        Args:
            graph dict[dict]: Each key is index of a road segment.
                            Corresponding value is dict containing child road segment and corresponding weight.
            spawn_points [str]: Indices corresponding to spawn points in the simulation.
            goal_points [str]: Indices corresponding to target positions in the simulation.
            routes [tuple]: Pairs of valid combinations of spawn and goal points.
        """
        self.graph = graph  # Road connectivity graph
        self.spawn_points = spawn_points  # Spawn road segments
        self.goal_points = goal_points  # Goal road segments
        self.start_end = routes  # Valid pairs of start and end points
        self.routes = [shortest_path(self.graph, route[0], route[1]) for route in routes]
        roads = self.road_data()
        self.segments = [Bezier2(road) for road in [*roads[0], *roads[1]]]

        for lane in roads[2]:
            points = lane['points']
            a, b = lane['ab']
            h, k = lane['centre']
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(7) % len(points)], points[(8) % len(points)] % (2 * pi)],
                                          colour_index=7))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(0) % len(points)], points[(1) % len(points)]],
                                          colour_index=0))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=(points[(1) % len(points)], points[(2) % len(points)]),
                                          colour_index=1))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(2) % len(points)], points[(3) % len(points)] + 2 * pi],
                                          colour_index=2))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(3) % len(points)], points[(4) % len(points)]],
                                          colour_index=3))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(4) % len(points)], points[(5) % len(points)]],
                                          colour_index=4))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(5) % len(points)], points[(6) % len(points)]],
                                          colour_index=5))
            self.segments.append(Ellipse2(centre=(h, k), r=(a, b),
                                          points=[points[(6) % len(points)], points[(7) % len(points)]],
                                          colour_index=6))

    @staticmethod
    def road_data(left=True):
        waypoints = {
            0: {
                'pos': (0.4751 * scene.WINDOW_WIDTH, 0),
                },
            1: {
                'pos': (0.4901 * scene.WINDOW_WIDTH, 0),
                },
            2: {
                'pos': (0.5201 * scene.WINDOW_WIDTH, 0),
                },
            3: {
                'pos': (0.5351 * scene.WINDOW_WIDTH, 0),
                },
            4: {
                'pos': (scene.WINDOW_WIDTH, 0.477 * scene.WINDOW_HEIGHT),
                },
            5: {
                'pos': (scene.WINDOW_WIDTH, 0.505 * scene.WINDOW_HEIGHT),
                },
            6: {
                'pos': (scene.WINDOW_WIDTH, 0.555 * scene.WINDOW_HEIGHT),
                },
            7: {
                'pos': (scene.WINDOW_WIDTH, 0.585 * scene.WINDOW_HEIGHT),
                },
            8: {
                'pos': (0.51 * scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT),
                },
            9: {
                'pos': (0.495 * scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT),
                },
            10: {
                'pos': (0, 0.555 * scene.WINDOW_HEIGHT),
                },
            11: {
                'pos': (0, 0.525 * scene.WINDOW_HEIGHT),
                },
            12: {
                'pos': (0, 0.471 * scene.WINDOW_HEIGHT),
                'grad': []
                },
            13: {
                'pos': (0, 0.445 * scene.WINDOW_HEIGHT),
                },
            14: {
                'pos': (0.475 * scene.WINDOW_WIDTH, 0.31 * scene.WINDOW_HEIGHT),
                },
            15: {
                'pos': (0.49 * scene.WINDOW_WIDTH, 0.31 * scene.WINDOW_HEIGHT),
                },
            16: {
                'pos': (0.52 * scene.WINDOW_WIDTH, 0.31 * scene.WINDOW_HEIGHT),
                },
            17: {
                'pos': (0.535 * scene.WINDOW_WIDTH, 0.31 * scene.WINDOW_HEIGHT),
                },
            18: {
                'pos': (0.68 * scene.WINDOW_WIDTH, 0.477 * scene.WINDOW_HEIGHT),
                },
            19: {
                'pos': (0.68 * scene.WINDOW_WIDTH, 0.505 * scene.WINDOW_HEIGHT),
                },
            20: {
                'pos': (0.7 * scene.WINDOW_WIDTH, 0.555 * scene.WINDOW_HEIGHT),  # todo Gradient of the straight line must be >>0 (old height: 0.555) 0.540
                },
            21: {
                'pos': (0.7 * scene.WINDOW_WIDTH, 0.585 * scene.WINDOW_HEIGHT),  # todo Gradient of the straight line must be >>0 (old height: 0.585) 0.570
                },
            22: {
                'pos': (0.5101 * scene.WINDOW_WIDTH, 0.8 * scene.WINDOW_HEIGHT),
                },
            23: {
                'pos': (0.4951 * scene.WINDOW_WIDTH, 0.8 * scene.WINDOW_HEIGHT),
                },
            24: {
                'pos': (scene.WINDOW_WIDTH * 470 / 1280, 0.57 * scene.WINDOW_HEIGHT),
                },
            25: {
                'pos': (scene.WINDOW_WIDTH * 470 / 1280, 0.54 * scene.WINDOW_HEIGHT),
                },
            26: {
                'pos': (scene.WINDOW_WIDTH * 450 / 1280, 0.485 * scene.WINDOW_HEIGHT),
                },
            27: {
                'pos': (scene.WINDOW_WIDTH * 450 / 1280, 0.46 * scene.WINDOW_HEIGHT),
                },
        }

        def straight():
            # roads = [(waypoints[i]['pos'], waypoints[i+14]['pos']) for i in range(14)]
            ls = [0, 1, 4, 5, 8, 10, 11]
            if left:
                # If driving on the left
                roads = [(waypoints[i+14]['pos'], waypoints[i]['pos']) if ls.count(i) else (waypoints[i]['pos'], waypoints[i+14]['pos']) for i in range(14)]
            else:
                roads = [(waypoints[i]['pos'], waypoints[i+14]['pos']) if ls.count(i) else (waypoints[i+14]['pos'], waypoints[i]['pos']) for i in range(14)]

            for i in range(14):
                j = i + 14
                start = waypoints[i]['pos']
                end = waypoints[j]['pos']
                grad = (end[1] - start[1]) / (end[0] - start[0])
                waypoints[i]['grad'] = grad
                waypoints[j]['grad'] = grad
            return roads

        def roundabout(centre, radii, points):
            rabout = [{
                'centre': centre,
                'ab': (r * scene.WINDOW_WIDTH / 1280, r * scene.WINDOW_HEIGHT / 720),
                'points': points
            } for r in radii]
            return rabout

        roads = straight()
        # points = [pi / 6, pi / 3, pi * 2 / 3, pi * 19 / 24, pi * 7 / 6, pi * 15 / 12, pi * 5 / 3, pi * 21 / 12, pi * 13 / 6]
        # points = [pi * 15 / 12, pi * 5 / 3, pi * 21 / 12, pi * 13 / 6, pi * (2 + 1 / 3), pi * (2 + 2 / 3), pi * (2 + 19 / 24),  pi * (2 + 7 / 6)]
        points = [pi * 16 / 12, pi * 5 / 3, pi * 43 / 24, pi * 27 / 12, pi * (2 + 9 / 24), pi * (2 + 18 / 24), pi * (2 + 19 / 24),  pi * (2 + 29 / 24)]
        points = [p % (2 * pi) for p in points]
        roundabout = roundabout(centre=(0.508 * scene.WINDOW_WIDTH, 0.52 * scene.WINDOW_HEIGHT), radii=[122, 104], points=points + [points[0]+2*pi])

        # Create waypoints on the roundabout
        r = [Ellipse(centre=lane['centre'], r=lane['ab'], points=lane['points']) for lane in roundabout]

        def grad(ri, point):
            dx, dy = ri.get_derivative(point)
            return dy / dx

        d = [{k+len(waypoints) + i*len(points): {'pos': lane.get_position(point), 'grad': grad(lane, point)} for (k, point) in enumerate(lane.points)} for i, lane in enumerate(r)]
        waypoint = waypoints | d[0] | d[1]

        def quadratic():
            """Quadratic bezier segments.
            start: 14-27
            end: 28-43
            """
            indices = [
                (14, 28),
                (15, 36),
                (37, 16),
                (29, 17),
                (18, 30),
                (19, 38),
                (39, 20),
                (31, 21),
                (22, 32),
                (33, 23),
                (24, 34),
                (25, 42),
                (43, 26),
                (35, 27)
            ]
            if left:
                # If driving on the left
                indices = [(i[1], i[0]) for i in indices]
            quad_beziers = [(waypoint[i[0]]['pos'], Bezier2.quad_control_point(start=waypoint[i[0]], end=waypoint[i[1]]), waypoint[i[1]]['pos']) for i in indices]
            return quad_beziers

        quads = quadratic()
        segments = [roads, quads, roundabout]

        return segments

    def update_geometry(self):
        roads = self.road_data()
        [self.segments[i].update_geometry(road) for i, road in enumerate([*roads[0], *roads[1]])]

        for i, lane in enumerate(roads[2]):
            for j in range(len(lane['points']) - 1):
                k = len([*roads[0], *roads[1]]) + i*(len(lane['points']) - 1) + j  # k: takes values 28-43 inclusive
                self.segments[k].update_geometry(lane)

    def draw(self, display_surface):
        [segment.draw(display_surface) for segment in self.segments]

    def plot_graph(self):
        """Writes the connectivity graph to a .svg file."""
        road_network_graph = pgv.AGraph(self.graph, strict=True, directed=True)
        road_network_graph.layout(prog='neato')  # Set graph layout
        road_network_graph.draw("graph.svg")  # Draw graph


class Curve:
    def __init__(self, t_min, t_max):
        self.t_min, self.t_max = t_min, t_max
        self.length = self.get_length(a=t_min, b=t_max)  # Length of the curve

        # Approximate the curve using 100 points
        self.line = [self.get_position(t) for t in linspace(self.t_min, self.t_max, 100)]

        self.occupiers = []  # List of vehicle objects currently using the segment

    def get_position(self, t):
        pass

    def get_length(self, b: float, a: float = 0, n: int = 3):
        """Calculates the length of the curve using gaussian quadrature.
        Args:
            a: Lower bound for the integration.
            b: Upper bound for the integration.
            n: Number of points for the approximation (2 to 6, 10 or 64)
        """
        def integrand(t):
            dx, dy = self.get_derivative(t)
            try:
                root = sqrt(dx ** 2 + dy ** 2)
            except OverflowError:
                logger.warning(f"Encountered OverflowError calculating: sqrt({dx}**2 + {dy}**2). Set result = 10000")
                root = 10000
            return root

        return abs(gauss_quadrature(integrand=integrand, a=a, b=b, n=n))

    def compute_position(self, s: float):
        """Returns the t-value corresponding to the actual distance travelled along the curve.
        Args:
            s: The distance the object should be located assuming linear relationship.

        Process:
        X(s) is the arc length parameterisation of the curve,
        Y(t) is another parameterisation of the curve
        Yt = self.get_position(t)

        Integrate the speed to obtain s as a function g(t) of time.
        s = g(t) = integral(dY(T)/dt dT, t_min, t)
        This computes the arc length corresponding to time, t.
        When t = t_min, s = g(t_min) = 0
             t = t_max, s = g(t_max) = L
        dYdt(T) = self.get_derivative(T)

        However, need to solve inverse problem -> Given arc length s, find time t at which it occurs.
        t = g^-1(s)

        Define F(t) = g(t) - s
        Want to find t such that F(t) = 0

        Root finding problem -> can be solved using Newton-Rhapson method:

        t_i+1 = t_i - F(t_i) / F'(t_i)
        With F'(t) = dF/dt = dg/dt = abs(dY/dt)
        dY/dt = self.get_derivative

        Potential problem with the Newton method:
        Guaranteed to converge if F''(t) >= 0 for all t E [t_min, t_max]
        If f''(t) can be -ve, can obtain t values outside of domain

        Therefore, use Newton-Rhapson - Bisection hybrid method.
        Bisection is used if a t_i+1 is outside the root-finding bounds.
        This guarantees convergence on every iteration step.
        """
        L = self.length  # The length of the curve

        def f(t):
            gt = self.get_length(a=self.t_min, b=t)
            return gt - abs(s)

        def df(t):
            dx, dy = self.get_derivative(t)
            return sqrt(dx ** 2 + dy ** 2)

        # Newton-Rhapson method
        t = self.t_min + (s / L) * (self.t_max - self.t_min)  # Initial guess for t
        err = 0.001  # Tolerance
        F = 2 * err
        lower, upper = self.t_min, self.t_max  # Initialise bounds for bisection method
        i = 0
        while abs(F) > err:
            i += 1
            if i > 10:
                break

            F = f(t)
            dF = df(t) #or 1e-6  # Prevent ZeroDivisionError
            try:
                next_t = t - F / dF
            except ZeroDivisionError:
                next_t = 100000  # Handle case if gradient is zero

            # Use bisection if the Newton approximation is outside the root-bounding interval.
            if F > 0:
                upper = t
                if next_t <= lower:
                    t = 0.5 * (upper + lower)
                else:
                    t = next_t
            else:
                lower = t
                if next_t >= upper:
                    t = 0.5 * (upper + lower)
                else:
                    t = next_t
        return t

    def draw(self, display_surface):
        pygame.draw.aalines(display_surface, (255, 255, 0), closed=False, points=self.line)  # Draw the line on the window

    def plot2(self):
        """Plots the arc length as a function of t."""
        line = list(zip(*[(s / self.length, self.get_length(b=self.compute_position(s)) / self.length) for s in linspace(0, self.length, 100)]))
        line2 = list(zip(*[(t / self.t_max, self.get_length(b=t) / self.length) for t in linspace(self.t_min, self.t_max, 100)]))  # With t = s / self.length
        plt.plot(*line, 'b--', *line2, 'r')
        plt.ylabel('Normalised Arc Length')
        plt.xlabel('Parameter')
        plt.legend(['Reparameterisation by Arc Length', 't'])
        plt.show()

    def plot(self):
        """Plots the arc length as a function of t."""
        line = list(zip(*[(s / self.length, self.get_length(b=self.compute_position(s)) / self.length) for s in linspace(0, self.length, 100)]))
        line2 = list(zip(*[(t / self.t_max, self.get_length(b=t) / self.length) for t in linspace(self.t_min, self.t_max, 100)]))  # With t = s / self.length
        return line, line2
        # plt.plot(*line, 'b--', *line2, 'r')
        # plt.ylabel('Normalised Arc Length')
        # plt.xlabel('Parameter')
        # plt.legend(['Reparameterisation by Arc Length', 't'])
        # plt.show()

    def update_geometry(self):
        self.length = self.get_length(a=self.t_min, b=self.t_max)  # Length of the curve
        # Approximate the curve using 100 points
        self.line = [self.get_position(t) for t in linspace(self.t_min, self.t_max, 100)]


class Bezier2(Curve):
    """A class for constructing and performing operations on Bézier curves."""
    t_min, t_max = 0, 1  # From Bezier function definition

    def __init__(self, weights: list):
        self.w = weights  # List of (x, y) points that define the Bézier curve.
        super().__init__(self.t_min, self.t_max)

    def get_position(self, t: float) -> tuple[float, float]:
        """Returns the point on the Bézier curve at the location t."""
        xt, yt = [self.bezier(t, [self.w[i][k] for i in range(len(self.w))]) for k in range(2)]
        return xt, yt

    def get_derivative(self, t: float) -> tuple[float, float]:
        """Returns the derivative of the Bézier curve at the location t."""
        # dx, dy = [self.deriv_bezier(t, w) for w in self.w]
        dx, _ = self.deriv_bezier(t, [self.w[i][0] for i in range(len(self.w))])
        dy, _ = self.deriv_bezier(t, [self.w[i][1] for i in range(len(self.w))])
        # [dx, _], [dy, _] = [self.deriv_bezier(t, [self.w[i][k] for i in range(len(self.w))]) for k in range(2)]
        return dx, dy

    def get_second_derivative(self, t) -> tuple[float, float]:
        dx, weights = self.deriv_bezier(t, [self.w[i][0] for i in range(len(self.w))])
        dx2, _ = self.deriv_bezier(t, [weights[i][0] for i in range(len(weights))])
        dy, weights = self.deriv_bezier(t, [self.w[i][1] for i in range(len(self.w))])
        dy2, _ = self.deriv_bezier(t, [weights[i][1] for i in range(len(weights))])
        return dx2, dy2

    def get_curvature(self, t: float) -> float:
        dx, dy = self.get_derivative(t)
        dx2, dy2 = self.get_second_derivative(t)
        kappa = (dx * dy2 - dx2 * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        return kappa

    def update_geometry(self, weights):
        self.w = weights  # List of (x, y) points that define the Bézier curve.
        super().update_geometry()

    def draw(self, display_surface):
        colours = [(127, 0, 255), (255, 127, 0)]
        pygame.draw.aalines(display_surface, colours[len(self.w) - 2], closed=False, points=self.line)  # Draw the line on the window

    @staticmethod
    def bezier(t: float, w: list):
        """A function to produce Bézier curves of any order (determined by len(w).
        Args:
            t: Parameter 0-1 at which to calculate (x, y)
            w: Weights (The desired coordinate values for the curve)
        """
        n = len(w) - 1
        sigma = sum([binomial(n, i) * (1 - t) ** (n - i) * t ** i * w[i] for i in range(n + 1)])
        return sigma

    @staticmethod
    def deriv_bezier(t: float, w: list) -> [float, list]:
        n = len(w) - 1
        k = n - 1
        sigma = sum([binomial(k, i) * (1 - t) ** (k - i) * t ** i * n * (w[i+1] - w[i]) for i in range(k+1)])
        weights = [n * (w[i+1] - w[i]) for i in range(k+1)]
        return sigma, weights

    @staticmethod
    def quad_control_point(start, end):
        """Calculates the position of the control point for a quadratic Bézier curve."""
        """
        Want to find a point (x, y) at the intercept of the tangents.
        Known: (x0, y0), (x1, y1), m0, m1 
        m0 = (y0 - y) / (x0 - x)
        m1 = (y - y1) / (x - x1)
        
        y0 - y = m0 (x0 - x)
        y = y0 - m0 (x0 - x) <- (1)
        
        y - y1 = m1 (x - x1)
        y = m1 (x - x1) + y1 <- (2)
        
        Equate (1) and (2)
        y0 - m0 (x0 - x) = m1 (x - x1) + y1
        y0 - y1 = m0 (x0 - x) + m1 (x - x1)
        y0 - y1 = x (m1 - m0) + m0 x0 - m1 x1
        
        
        x = (y0 - y1 - m0 x0 + m1 x1) / (m1 - m0)
        y = y0 - m0 (x0 - x)
        assert y == m1 (x - x1) + y1  # Check we found the correct point 
        return x, y       
        """
        x0, y0 = start['pos']
        m0 = start['grad']
        x1, y1 = end['pos']
        m1 = end['grad']

        x = (y0 - y1 - m0 * x0 + m1 * x1) / (m1 - m0)
        y = y0 - m0 * (x0 - x)
        # assert y == m1 * (x - x1) + y1  # Check we found the correct point
        return x, y


class Ellipse2(Curve):
    # t_min, t_max = 0, -2*pi  # From parametric ellipse definition (-2pi sets anticlockwise direction)
    colours = list(product([255, 0], repeat=3))  # Assign each segment a different colour
    direction = 1

    def __init__(self, centre: tuple, r: tuple, points: [float], colour_index: [int] = 0):
        self.h, self.k = centre  # Centre position
        self.a, self.b = r  # Major and minor semi-axis

        # y-axis gets inverted if travelling anticlockwise
        if self.direction == -1:
            self.points = [2 * pi - p for p in points]
        else:
            self.points = points

        self.start, self.end = self.points
        # self.t_min, self.t_max = self.start, self.end
        super().__init__(self.start, self.end)
        self.t_min, self.t_max = 0, 1

        if self.end < self.start:  # Ensure arc length gets calculated in correct direction
            self.end += 2*pi

        self.length = super().get_length(a=self.start, b=self.end)
        self.colour = self.colours[colour_index]

    def get_position(self, t: float):
        """Returns (x, y) coordinate on the ellipse at angle t.
        Args:
            t: Angle in range 0 to 2pi (radians)
        """
        t = self.convert_t(t)  # Convert t from range 0-1 to start-end angle
        x = self.h + self.a * cos(t % (2*pi))
        y = self.k + self.b * sin(t % (2*pi)) * self.direction
        return x, y

    def get_derivative(self, t: float):
        """Returns the derivative of the ellipse at the location t."""
        t = self.convert_t(t)  # Convert t from range 0-1 to start-end angle
        try:
            dx = self.a * -sin(t - self.start)
        except ValueError:
            dx = 0
            logger.warning(f"Encountered math domain error with dx={self.a} * -sin{t}")

        try:
            dy = self.b * cos(t - self.start) * self.direction  # Have to modify the start and end t value for each segment.
        except ValueError:
            dy = 0
            logger.warning(f"Encountered math domain error with dy={self.b} * cos{t}")

        return dx, dy

    def get_derivative2(self, t: float):
        """Returns the derivative of the ellipse at the location t."""
        t = self.convert_t(t)  # Convert t from range 0-1 to start-end angle
        try:
            dx = self.a * -sin(t)
        except ValueError:
            dx = 0
            logger.warning(f"Encountered math domain error with dx={self.a} * -sin{t}")

        try:
            dy = self.b * cos(t) * self.direction
        except ValueError:
            dy = 0
            logger.warning(f"Encountered math domain error with dy={self.b} * cos{t}")

        return dx, dy

    def get_curvature(self, t):
        dx, dy = self.get_derivative(t)
        kappa = (self.a * self.b) / ((dx ** 2 + dy ** 2) ** 1.5)
        return kappa

    def update_geometry(self, data):
        # self.h, self.k = data[0]  # Centre position
        # self.a, self.b = data[1]  # Major and minor semi-axis
        self.h, self.k = data['centre']  # Centre position
        self.a, self.b = data['ab']  # Major and minor semi-axis
        super().update_geometry()

    def convert_t(self, t):
        """
        Maps t from range 0-1 to range self.start-self.end
        """
        m = self.end - self.start
        return m * t + self.start

    def draw(self, display_surface):
        # line = [self.get_position(t) for t in linspace(self.start, self.end, 100)]
        pygame.draw.aalines(display_surface, self.colour, closed=False, points=self.line)  # Draw the line on the window


class Ellipse(Curve):
    t_min, t_max = 0, 2*pi  # From parametric ellipse definition
    colours = list(product([255, 0], repeat=3))  # Assign each segment a different colour

    def __init__(self, centre: tuple, r: tuple, points: [float]):
        self.h, self.k = centre  # Centre position
        self.a, self.b = r  # Major and minor semi-axis
        super().__init__(self.t_min, self.t_max)

        self.points = points
        # self.segments = [self.EllipseArc(self.points[i-1], self.points[i]) for i in range(len(self.points))]
        # for segment in self.segments:
        #     segment.length = super().get_length(a=segment.start, b=segment.end)

    def get_position(self, t: float):
        """Returns (x, y) coordinate on the ellipse at angle t.
        Args:
            t: Angle in range 0 to 2pi (radians)
        """
        x = self.h + self.a * cos(t)
        y = self.k + self.b * sin(t)
        return x, y

    def get_derivative(self, t: float):
        """Returns the derivative of the ellipse at the location t."""
        dx = self.a * -sin(t)
        dy = self.b * cos(t)
        return dx, dy

    def get_curvature(self, t):
        dx, dy = self.get_derivative(t)
        kappa = (self.a * self.b) / ((dx ** 2 + dy ** 2) ** 1.5)
        return kappa

    def update_geometry(self, data):
        self.h, self.k = data[0]  # Centre position
        self.a, self.b = data[1]  # Major and minor semi-axis
        super().update_geometry()

    def draw(self, display_surface):
        # Approximate the curve using 100 points
        lines = [[self.get_position(t) for t in linspace(self.points[i-1], self.points[i], 100)] for i in range(len(self.points))]
        # Draw the line on the window
        [pygame.draw.aalines(display_surface, colour, closed=False, points=line) for colour, line in zip(self.colours, lines)]


# =================================
# Functions
# =================================

def gauss_quadrature(integrand, a: float = 0, b: float = 1, n: int = 3):
    """Approximates the integral of a function using gaussian quadrature.
    Args:
        integrand callable: The function to be integrated.
        a float: Lower bound for the integration.
        b float: Upper bound for the integration.
        n int: Number of points for the approximation (2 to 5)
    """
    lut = {
        2: {
            'points': [-1 / sqrt(3), 1 / sqrt(3)],
            'weights': [1, 1]
        },
        3: {
            'points': [0, -sqrt(3 / 5), sqrt(3 / 5)],
            'weights': [8 / 9, 5 / 9, 5 / 9]
        },
        4: {
            'points': [-sqrt(3/7 - 2/7 * sqrt(6/5)), sqrt(3/7 - 2/7 * sqrt(6/5)), -sqrt(3/7 + 2/7 * sqrt(6/5)),
                       sqrt(3/7 + 2/7 * sqrt(6/5))],
            'weights': [(18 + sqrt(30)) / 36, (18 + sqrt(30)) / 36, (18 - sqrt(30)) / 36, (18 - sqrt(30)) / 36]
        },
        5: {
            'points': [0, -1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)), 1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)),
                       -1 / 3 * sqrt(5 + 2 * sqrt(10 / 7)), 1 / 3 * sqrt(5 + 2 * sqrt(10 / 7))],
            'weights': [128 / 225, (322 + 13 * sqrt(70)) / 900, (322 + 13 * sqrt(70)) / 900, (322 - 13 * sqrt(70)) / 900,
                        (322 - 13 * sqrt(70)) / 900]
        },
        6: {
            'points': [0.6612093864662645, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969,
                       -0.9324695142031521, 0.9324695142031521],
            'weights': [0.3607615730481386, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910,
                        0.1713244923791704, 0.1713244923791704]
        },
        10: {
            'points': [-0.1488743389816312, 0.1488743389816312, -0.4333953941292472, 0.4333953941292472,
                       -0.6794095682990244, 0.6794095682990244, -0.8650633666889845, 0.8650633666889845,
                       -0.9739065285171717, 0.9739065285171717],
            'weights': [0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2692667193099963,
                        0.2190863625159820, 0.2190863625159820, 0.1494513491505806, 0.1494513491505806,
                        0.0666713443086881, 0.0666713443086881]
        },
        64: {
            'points': [-0.0243502926634244, 0.0243502926634244, -0.0729931217877990, 0.0729931217877990,
                       -0.1214628192961206, 0.1214628192961206, -0.1696444204239928, 0.1696444204239928,
                       -0.2174236437400071, 0.2174236437400071, -0.2646871622087674, 0.2646871622087674,
                       -0.3113228719902110, 0.3113228719902110, -0.3572201583376681, 0.3572201583376681,
                       -0.4022701579639916, 0.4022701579639916, -0.4463660172534641, 0.4463660172534641,
                       -0.4894031457070530, 0.4894031457070530, -0.5312794640198946, 0.5312794640198946,
                       -0.5718956462026340, 0.5718956462026340, -0.6111553551723933, 0.6111553551723933,
                       -0.6489654712546573, 0.6489654712546573, -0.6852363130542333, 0.6852363130542333,
                       -0.7198818501716109, 0.7198818501716109, -0.7528199072605319, 0.7528199072605319,
                       -0.7839723589433414, 0.7839723589433414, -0.8132653151227975, 0.8132653151227975,
                       -0.8406292962525803, 0.8406292962525803, -0.8659993981540928, 0.8659993981540928,
                       -0.8893154459951141, 0.8893154459951141, -0.9105221370785028, 0.9105221370785028,
                       -0.9295691721319396, 0.9295691721319396, -0.9464113748584028, 0.9464113748584028,
                       -0.9610087996520538, 0.9610087996520538, -0.9733268277899110, 0.9733268277899110,
                       -0.9833362538846260, 0.9833362538846260, -0.9910133714767443, 0.9910133714767443,
                       -0.9963401167719553, 0.9963401167719553, -0.9993050417357722, 0.9993050417357722],
            'weights': [0.0486909570091397, 0.0486909570091397, 0.0485754674415034, 0.0485754674415034,
                        0.0483447622348030, 0.0483447622348030, 0.0479993885964583, 0.0479993885964583,
                        0.0475401657148303, 0.0475401657148303, 0.0469681828162100, 0.0469681828162100,
                        0.0462847965813144, 0.0462847965813144, 0.0454916279274181, 0.0454916279274181,
                        0.0445905581637566, 0.0445905581637566, 0.0435837245293235, 0.0435837245293235,
                        0.0424735151236536, 0.0424735151236536, 0.0412625632426235, 0.0412625632426235,
                        0.0399537411327203, 0.0399537411327203, 0.0385501531786156, 0.0385501531786156,
                        0.0370551285402400, 0.0370551285402400, 0.0354722132568824, 0.0354722132568824,
                        0.0338051618371416, 0.0338051618371416, 0.0320579283548516, 0.0320579283548516,
                        0.0302346570724025, 0.0302346570724025, 0.0283396726142595, 0.0283396726142595,
                        0.0263774697150547, 0.0263774697150547, 0.0243527025687109, 0.0243527025687109,
                        0.0222701738083833, 0.0222701738083833, 0.0201348231535302, 0.0201348231535302,
                        0.0179517157756973, 0.0179517157756973, 0.0157260304760247, 0.0157260304760247,
                        0.0134630478967186, 0.0134630478967186, 0.0111681394601311, 0.0111681394601311,
                        0.0088467598263639, 0.0088467598263639, 0.0065044579689784, 0.0065044579689784,
                        0.0041470332605625, 0.0041470332605625, 0.0017832807216964, 0.0017832807216964]
        }
    }

    points, w = lut[n].values()
    sigma = sum([w * integrand(0.5 * (b - a) * point + 0.5 * (a + b)) for point, w in zip(points, w)])
    return 0.5 * (b - a) * sigma


def linspace(start: float, stop: float, n: int) -> float:
    """Generator that returns n linearly spaced floats in the range (start, stop)."""
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i


def binomial(n, k):
    """Pascal's triangle"""
    lut = [[1],  # n = 0
           [1, 1],  # n = 1
           [1, 2, 1],  # n = 2
           [1, 3, 3, 1],  # n = 3
           [1, 4, 6, 4, 1],  # n = 4
           [1, 5, 10, 10, 5, 1],  # n = 5
           [1, 6, 15, 20, 15, 6, 1]]  # n = 6

    # Expand lut if required (n, k) pair doesn't exist yet.
    while n >= len(lut):
        prev = len(lut) - 1
        next_row = [lut[prev][i] + lut[prev][i+1] for i in range(prev)]
        next_row = [1] + next_row + [1]
        lut.append(next_row)

    return lut[n][k]


def main():
    world = World()  # Create the Pygame simulation environment

    graph = {
        '0': {'14': 1},
        '1': {'15': 1},
        '2': {},
        '3': {},
        '4': {'18': 1},
        '5': {'19': 1},
        '6': {},
        '7': {},
        '8': {'22': 1},
        '9': {},
        '10': {'24': 1},
        '11': {'25': 1},
        '12': {},
        '13': {},
        '14': {'28': 1},
        '15': {'36': 1},
        '16': {'2': 1},
        '17': {'3': 1},
        '18': {'30': 1},
        '19': {'38': 1},
        '20': {'6': 1},
        '21': {'7': 1},
        '22': {'32': 1},
        '23': {'9': 1},
        '24': {'34': 1},
        '25': {'42': 1},
        '26': {'12': 1},
        '27': {'13': 1},
        '28': {'27': 1, '35': 1},
        '29': {'28': 1},
        '30': {'17': 1, '29': 1},
        '31': {'30': 1},
        '32': {'21': 1, '31': 1},
        '33': {'32': 1},
        '34': {'23': 1, '33': 1},
        '35': {'34': 1},
        '36': {'26': 1, '43': 1},
        '37': {'36': 1},
        '38': {'16': 1, '37': 1},
        '39': {'38': 1},
        '40': {'20': 1, '39': 1},
        '41': {'40': 1},
        '42': {'41': 1},
        '43': {'42': 1}
    }  # Adjacency matrix
    goal_points = ['0', '1', '4', '5', '8', '10', '11']
    spawn_points = ['2', '3', '6', '7', '9', '12', '13']
    routes = [
        ('0', '13'),
        ('0', '9'),
        ('1', '6'),
        ('1', '2'),
        ('4', '3'),
        ('4', '13'),
        ('5', '6'),
        ('8', '7'),
        ('8', '3'),
        ('10', '9'),
        ('10', '7'),
        ('11', '2'),
        ('11', '12'),
    ]
    # routes = [
    #     ('0', '13')
    #     # ('0', '9'),
    #     # ('1', '6'),
    #     # ('1', '2'),
    #     # ('4', '3'),
    #     # ('4', '13'),
    #     # ('5', '6'),
    #     # ('8', '7'),
    #     # ('8', '3'),
    #     # ('10', '9'),
    #     # ('10', '7'),
    #     # ('11', '2'),
    #     # ('11', '12'),
    # ]

    routes = [(r[1], r[0]) for r in routes]  # Valid pairs of spawn/goal segments
    # Driving on the left ----
    graph = TD(graph, 1)
    v = {str(i): {} for i in range(44) if str(i) not in graph.keys()}
    graph = v | graph
    # -----

    # todo The road network must be available to both traffic and the agent
    world.TM = TrafficManager(RoadNetwork(graph=graph, spawn_points=spawn_points, goal_points=goal_points, routes=routes))  # Add traffic to the simulation

    world.road_network = RoadNetwork(graph=graph, spawn_points=spawn_points, goal_points=goal_points, routes=routes)
    world.road_network.plot_graph()  # Draw road connectivity graph

    def plot_arc_length():
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharex="all", sharey="all")
        for i in range(15, 30):  # 16, 30 for bezier
            segment = world.road_network.segments[i]
            line, line2 = segment.plot()
            ax[int(i >= 16)].plot(*line2, 'r', linewidth='0.5')

        [ax[i].plot(*line, 'b--', linewidth='1') for i in range(2)]

        ax[0].title.set_text("Ellipse")
        ax[1].title.set_text("Quadratic Bezier")
        ax[0].set_ylabel('Normalised Arc Length')
        [ax[i].set_xlabel('Normalised Parameter') for i in range(2)]
        ax[0].legend(['t', 'Reparameterisation by Arc Length'])
        ax[0].set_xlim([0, 1])
        ax[0].set_ylim([0, 1])
        [ax[i].set_box_aspect(1) for i in range(2)]

        plt.show()

    # plot_arc_length()

    try:
        world.game_loop(live_data=True)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
