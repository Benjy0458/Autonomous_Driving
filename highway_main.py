# =================================
# Imports
# =================================
import os
import pygame
import time
import random
import threading
import sys
import curses

from behaviour_executive import finite_state_machine as fsm
from behaviour_executive import behaviour_tree as bt
from path_planner.hybrid_astar import HybridAstar
from path_planner.normal_astar import Astar
from utils import RepeatedTimer

from multiprocessing import Process, Queue
from scipy import interpolate as itp
import numpy as np
from math import ceil, sin, cos, pi, sqrt

from controllers import IDM, OptimalControl

from scenario.highway import LANES, FPS, LANE_VELOCITIES
import scenario.highway as scene
import plotting

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler(f'LogFiles/Highway/{__name__}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger

# =================================
# Global variables
# =================================
TIME_GAIN = 1  # Speed up the simulation - Doesn't affect path planner or BT subprocesses
FPS /= TIME_GAIN

TRAFFIC_DENSITY = 2  # Density of NPC vehicles in the simulation


# =================================
# Classes
# =================================


class World:
    """A class that handles the pygame simulation environment"""

    def __init__(self):
        self.display_surface = World.init_pygame()  # Open a new named Pygame window
        image = pygame.image.load(rf'{scene.BACKGROUND_IMAGE}').convert()
        self.image = pygame.transform.scale(image,
                                            (scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT))  # Scale the background image

        self.display_surface.blit(self.image, (0, 0))  # Fill the pygame window with highway image

        self.TM = None  # Create a new instance of the traffic manager
        self.agent = None  # Create the agent
        self.start_time = 0  # Stores the start time of the simulation
        logger.info(f'Created new instance of the simulation environment.')

    def game_loop(self, live_data: bool = False) -> None:
        self.start_time = time.time()  # Store the start time of the simulation
        clock = pygame.time.Clock()  # Controls the frame rate

        if live_data:
            i = 0  # A counter that determines when to send data to the live plot
            queue = Queue()  # Stores data for the live graph
            close = Queue()  # Send data to exit the live graph
            live_plot = Process(target=plotting.live_graph, args=(queue, close))
            live_plot.start()

        self.TM and self.TM.timed_spawn.start()  # Spawn vehicles at regular intervals
        self.agent and self.agent.bt_tick.start()  # Start the agent behaviour tree subprocess
        self.agent and self.agent.path_planner.start()  # Start the agent path planner subprocess

        running = True
        try:
            logger.info(f'Simulation started.')
            pygame.event.set_allowed(pygame.QUIT)  # Set which events are monitored by Pygame
            while running:
                for event in pygame.event.get():  # Decide whether to terminate the simulation
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            pygame.image.save(self.display_surface, f"{time.time()}.jpeg")

                elapsed_time = time.time() - self.start_time  # Calculate elapsed time
                global TRAFFIC_DENSITY
                # TRAFFIC_DENSITY = 5 * sin(0.1 * elapsed_time) + 5
                # TRAFFIC_DENSITY = 10 * sin(0.1 * elapsed_time) ** 0.5
                TRAFFIC_DENSITY = -sin(0.2 * elapsed_time + 0.5 * pi) + 2 * sin(0.05 * elapsed_time - 0.5 * pi) + 3

                self.TM and self.update_vehicles()  # Update NPC vehicles
                self.agent and self.update_agent()  # Update and draw the agent

                self.draw_window()  # Draw objects on the pygame window
                pygame.display.update()  # Update the pygame window
                self.caption(elapsed_time, clock)  # Update the window caption

                if live_data and self.agent:
                    i = (i + 1) % 4  # plot data every 4th frame
                    if not i:
                        queue.put((elapsed_time, self.agent.xte, self.agent.phi, self.agent.delta, self.agent.velocity, self.agent.acc))

                clock.tick_busy_loop(FPS * TIME_GAIN)  # Ensure program maintains desired frame rate
        finally:
            if self.TM:
                self.TM.timed_spawn.stop()  # Stop the spawn vehicles thread
                logger.debug('Successfully terminated the spawn vehicles thread.')
            if self.agent:
                self.agent.path_planner.kill()  # Stop the path planner subprocess
                logger.debug('Successfully terminated the path planner subprocess.')
                self.agent.bt_tick.kill()  # Stop the BT process
                logger.debug('Successfully terminated the behaviour tree subprocess.')
                logger.info(f'Simulation terminated successfully: {self.agent.terminal_count} '
                            f'scenarios simulated with {self.agent.n_hits} collisions.')

            if live_data:
                close.put(1)
                live_plot.join(timeout=5)
                live_plot.kill()
                logger.debug("Succesfully closed the live plot.")

    def caption(self, elapsed_time: float, clock: pygame.time.Clock) -> None:
        tracks_count = self.TM.lane_distribution() if self.TM else 0
        cap = []
        sim_cap = f"Elapsed time: {round(elapsed_time)}, FPS: {round(clock.get_fps())}, " \
                  f"Traffic density: {round(TRAFFIC_DENSITY, 1)}"
        cap.append(sim_cap)
        if self.TM:
            tm_cap = f"Num Vehicles: {sum(tracks_count)}, Lane dist: {tracks_count}"
            cap.append(tm_cap)
        if self.agent:
            agent_cap = f"Scenarios simulated: {self.agent.terminal_count}, Collisions: {self.agent.n_hits}, " \
                        f"Speed: {round(self.agent.velocity)} mph, Lane: {self.agent.lane}"
            cap.append(agent_cap)

        caption = ", ".join(cap)
        pygame.display.set_caption(caption)

    def update_vehicles(self) -> None:
        """Updates the positions of all traffic on the screen. Removes vehicles that have moved off-screen."""
        # [car.update_position() if 0 <= car.x_pos <= scene.WINDOW_WIDTH else self.TM.vehicles.remove(car)
        #  for car in self.TM.vehicles]
        [car.update_position() if 0 <= car.x_pos <= scene.WINDOW_WIDTH else self.TM.ordered_vehicles[i].remove(car)
         for i, lane in enumerate(self.TM.ordered_vehicles) for car in lane]

    def update_agent(self) -> None:
        self.agent.update_position(self.TM and self.TM.ordered_vehicles)  # Update the agent

    def draw_window(self) -> None:
        def draw_radar(horizontal: bool = False) -> None:
            [pygame.draw.aaline(self.display_surface, (255, 0, 0),
                                (self.agent.x_pos, LANES[car.lane] if horizontal else self.agent.y_pos),
                                (car.x_pos, LANES[car.lane]))
             for car in self.agent.sensor.radar.closest_cars if car and car.lane > 2]

        self.display_surface.blit(self.image, (0, 0))  # Fill the pygame window with the background image
        self.agent and draw_radar(horizontal=False)
        # Draw traffic
        # self.TM and [pygame.draw.rect(self.display_surface, car.colour, car.rect) for car in self.TM.vehicles]
        self.TM and [pygame.draw.rect(self.display_surface, car.colour, car.rect) for lane in self.TM.ordered_vehicles
                     for car in lane]
        if self.agent:
            # Draw path
            (len(self.agent.path) > 1 and pygame.draw.aalines(
             self.display_surface, (250, 0, 250), False, [(k, v[0]) for k, v in self.agent.path.items()]))

            pygame.draw.rect(self.display_surface, self.agent.colour, self.agent.rect)  # Draw agent
            # Draw deadzone, collisions detected before this line are not counted
            pygame.draw.line(self.display_surface, (0, 255, 255),
                             (scene.DEAD_ZONE, 0), (scene.DEAD_ZONE, scene.WINDOW_HEIGHT))

    @staticmethod
    def init_pygame() -> pygame.Surface:
        """Returns a new named pygame window."""
        # global WINDOW_WIDTH, WINDOW_HEIGHT
        x_pos, y_pos = 0, 30
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x_pos, y_pos)  # Set the position of the window on the screen
        pygame.init()
        if not scene.WINDOW_WIDTH:
            scene.WINDOW_WIDTH = pygame.display.Info().current_w
        if not scene.WINDOW_HEIGHT:
            scene.WINDOW_HEIGHT = 120
        ds = pygame.display.set_mode((scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT))
        pygame.display.set_caption('Autonomous Highway Driving')

        return ds


class TrafficManager:
    """A class that handles the behaviour of NPC vehicles in the simulation"""

    def __init__(self):
        self.vehicles = []  # Keeps track of the vehicles currently in the window
        self.ordered_vehicles = [[], [], [], [], [], []]
        self.timed_spawn = RepeatedTimer(1 / (TRAFFIC_DENSITY * TIME_GAIN), self.spawn_vehicles)
        logger.info('Created new instance of the traffic manager.')

    def spawn_vehicles(self):
        """Creates a new vehicle and appends it to the list of vehicles."""
        max_vehicles = 3
        new_vehicles = []  # Temporary list to store our new vehicles.
        number_of_vehicles = random.randint(0, max_vehicles)  # Assign a random number of vehicles to spawn in a range.
        vehicle_categories = random.choices(population=["car", "truck"], cum_weights=[0.7, 1.0], k=number_of_vehicles)

        for v_cat in vehicle_categories:
            new_vehicle = Vehicle(vehicle_type=v_cat)  # Create a new vehicle
            for vehicle in new_vehicles:
                if new_vehicle.lane == vehicle.lane:
                    break
            else:
                if self.ordered_vehicles[new_vehicle.lane]:
                    # Get the last vehicle spawned in new_vehicle's lane (direction independent)
                    new_vehicle.front_car = self.ordered_vehicles[new_vehicle.lane][-1]
                    # Check if the new_vehicle is colliding with the front vehicle or agent
                    collide = new_vehicle.rect.collidelist([new_vehicle.front_car.rect, world.agent.rect]) \
                        if world.agent else new_vehicle.rect.collidelist([new_vehicle.front_car.rect])
                    if collide != -1:
                        break

                self.ordered_vehicles[new_vehicle.lane].append(new_vehicle)

    def lane_distribution(self):
        """Returns the number of vehicles in each lane."""
        # tracks = [vehicle.lane for vehicle in self.vehicles]
        # tracks_count = [tracks.count(track) for track in range(6)]
        # tracks_count = [len(cars) for cars in self.ordered_vehicles]
        return [len(cars) for cars in self.ordered_vehicles]


class Vehicle:
    """Car class for NPC vehicles"""
    scale = 10 / 3
    lane_bias = 0.5  # The probability of a vehicle spawning in the top three lanes.

    def __init__(self, vehicle_type: str = "car"):
        match vehicle_type:
            case "car":
                # Geometric parameters and initial lane
                self.width = random.normalvariate(2, 0.15) * self.scale  # Width of the vehicle (y-direction)
                self.length = random.normalvariate(4.5, 0.2) * self.scale  # Length of the vehicle (x-direction)
                # Choose a random lane for the vehicle
                self.lane = random.choice([3, 4, 5]) if random.random() > self.lane_bias else random.choice([0, 1, 2])
                self.driving_style = random.choices(population=["safe", "aggressive"], weights=[0.2, 0.8])[0]
                self.colour = pygame.Color(0, 0, 255) if self.driving_style == "aggressive" else pygame.Color(0, 255, 0)
            case "truck":
                # Geometric parameters and initial lane
                self.width = random.normalvariate(2.1, 0.1) * self.scale  # Width of the vehicle (y-direction)
                self.length = random.normalvariate(16.0, 2) * self.scale  # For trucks
                # Trucks won't spawn in the fast lane
                self.lane = random.choice([4, 5]) if random.random() > self.lane_bias else random.choice([0, 1])
                self.driving_style = "wide_load"
                self.colour = pygame.Color(255, 0, 0)

        self.direction = 1 if self.lane > 2 else -1
        self.front_car = None  # The vehicle object of the car in front

        # Kinematic parameters
        self.acc = 0  # The initial acceleration of the vehicle
        self.max_velocity = 1.1 * LANE_VELOCITIES[self.lane] if self.driving_style == "aggressive" else LANE_VELOCITIES[
            self.lane]
        self.x_velocity = LANE_VELOCITIES[self.lane]  # x velocity of the vehicle
        self.y_velocity = 0

        # Set spawn location
        self.y_pos = LANES[self.lane]  # The y-position of the vehicle's lane
        self.x_pos = 0 if self.direction == 1 else scene.WINDOW_WIDTH  # Set start x pos depending on travel direction

        # Assign the vehicle a random colour
        # self.colour = pygame.Color(int(random.random() * 255), int(random.random() * 255), int(random.random() * 255))

        self.rect = pygame.Rect(0, 0, self.length, self.width)  # Create the vehicle's rectangle object
        self.rect.center = (self.x_pos, self.y_pos)

        # Initialise speed controller
        self.speed_controller = IDM(driving_style="safe", desired_velocity=self.max_velocity)

    def update_position(self):
        """
        Agent and front vehicle: Front car is the closest vehicle
        Agent only: Front car is agent
        Front vehicle only: Front car is front car
        Free road: Arbitrarily large distance and desired velocity
        """
        if world.agent and world.agent.lane == self.lane and world.agent.x_pos > self.x_pos:
            agent_car_distance = world.agent.x_pos - self.x_pos - 0.5 * (self.length + world.agent.length)
            if self.front_car and scene.WINDOW_WIDTH > self.front_car.x_pos > 0:
                front_car_distance = self.direction * (self.front_car.x_pos - self.x_pos) - 0.5 * (
                        self.length + self.front_car.length)
                if agent_car_distance < front_car_distance:
                    acc = self.speed_controller.speed_control.send([self.x_velocity, agent_car_distance,
                                                                    world.agent.velocity]) * 2.237  # Update agent velocity in mph/s
                else:
                    acc = self.speed_controller.speed_control.send([self.x_velocity, front_car_distance,
                                                                    self.front_car.x_velocity]) * 2.237  # Update agent velocity in mph/s
            else:
                acc = self.speed_controller.speed_control.send([self.x_velocity, agent_car_distance,
                                                                world.agent.velocity]) * 2.237  # Update agent velocity in mph/s
        elif self.front_car and scene.WINDOW_WIDTH > self.front_car.x_pos > 0:
            front_car_distance = self.direction * (self.front_car.x_pos - self.x_pos) - 0.5 * (
                    self.length + self.front_car.length)
            acc = self.speed_controller.speed_control.send([self.x_velocity, front_car_distance,
                                                            self.front_car.x_velocity]) * 2.237  # Update agent velocity in mph/s
        else:
            acc = self.speed_controller.speed_control.send(
                [self.x_velocity, 1000, LANE_VELOCITIES[self.lane]]) * 2.237  # Update agent velocity in mph/s

        self.x_velocity += acc * (1 / FPS)

        self.x_pos += ((self.x_velocity * 1 / FPS) + 0.5 * acc * (1 / FPS) ** 2) * self.direction  # Increment position
        self.y_pos += self.y_velocity * 1 / FPS
        self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the vehicle rectangle


class Agent:
    """Car class for the Agent vehicle"""
    length = 15  # length (x-dim) of the car
    width = 8  # width (y-dim) of the car

    min_speed = 30  # Min speed of the agent car (mph)
    max_speed = scene.SPEED_LIMIT  # Max speed (mph)
    acceleration = 3  # Max acceleration of the agent (m/s^2)
    deceleration = 9.81  # Max deceleration (m/s^2)

    def __init__(self, path_planner: type[Astar or HybridAstar] = HybridAstar, speed_controller=IDM):
        self.lane = 5  # Initial lane
        self.colour = (0, 0, 0)  # Agent colour
        self.reset()  # Sets initial position and velocity
        # rectangle object for the car
        self.rect = pygame.Rect(self.x_pos - int(self.length / 2), self.y_pos - int(self.width / 2), self.length,
                                self.width)

        self.car_front1 = self.car_front2 = scene.WINDOW_WIDTH
        self.n_hits = self.successes = self.terminal_count = 0  # Counters (collisions, reached end of road, total)

        self.sensor = Sensor(self)  # Create the agent sensors
        self.fsm = fsm.HighwayDrive(self, self.sensor)  # Initialise finite state machine

        # Initialise behaviour tree
        self.bt_send, self.bt_return = Queue(), Queue()  # Queue objects to send/receive data from the BT
        self.bt_agent_data = self.BtData(self)
        self.bt_tick = Process(target=bt.setup, args=(self.bt_send, self.bt_return, bt.highway_drive))

        self.speed_controller = speed_controller(driving_style="safe",
                                                 desired_velocity=self.max_speed)  # Initialise speed controller

        # Initialise trajectory tracker
        self.lateral_controller = OptimalControl(self.length)

        # Initialise the path planner
        self.send_queue, self.return_queue = Queue(), Queue()  # Queue objects to send/receive data from path planner
        self.path_planner = Process(target=self.get_path, args=(path_planner, self.send_queue, self.return_queue))
        logger.info('Created new instance of the agent vehicle.')

    class BtData:
        """A class to data to send to the agent object."""

        def __init__(self, agent) -> None:
            """Initialise variables to arbitrary defaults."""
            self.initialise(agent)

        def initialise(self, agent) -> None:
            """Resets the data for the BT."""
            self.goal_pos = agent.goal_pos
            self.x_velocity = agent.velocity
            self.max_speed = agent.max_speed
            self.length = agent.length

            self.x_pos = agent.x_pos
            self.lane = agent.lane

            # Surrounding vehicles
            self.front_car_distance = agent.sensor.radar.distances[0]
            try:
                self.front_car_velocity = agent.sensor.radar.closest_cars[0].x_velocity
            except AttributeError:
                self.front_car_velocity = float('inf')

            self.front_right_car_distance = agent.sensor.radar.distances[4]
            self.rear_right_car_distance = agent.sensor.radar.distances[5]

            self.front_left_car_distance = agent.sensor.radar.distances[2]
            self.rear_left_car_distance = agent.sensor.radar.distances[3]
            try:
                self.front_left_car_velocity = agent.sensor.radar.closest_cars[2].x_velocity
            except AttributeError:
                self.front_left_car_velocity = float('inf')

            self.rear_car_distance = agent.sensor.radar.distances[1]
            try:
                self.rear_car_velocity = agent.sensor.radar.closest_cars[1].x_velocity
            except AttributeError:
                self.rear_car_velocity = float('inf')

        def __str__(self) -> str:
            return str(self.__dict__)

    def reset(self):
        """Reset the agent"""
        self.r = 0  # The distance travelled from the start position
        self.dr = 0  # The distance travelled from the previous timestep
        self.x_pos = 0
        self.y_pos = LANES[self.lane]
        self.traj_y_pos = self.y_pos  # The y-position of the current path
        self.traj_heading = 0  # The angle of the current path
        self.xte = 0  # Cross-track error
        self.phi = 0  # Heading error
        self.delta = 0  # The current steering angle
        self.theta = 0  # The current heading angle
        self.acc = 0  # The longitudinal acceleration
        self.velocity = LANE_VELOCITIES[self.lane]
        self.goal_pos = [self.x_pos + 40, self.y_pos]  # Goal position for the path planner
        self.prev_goal_pos = [self.x_pos, self.y_pos]
        self.path = {}

    def update_position(self, vehicles):
        def perception():
            self.sensor.collision.observation(vehicles) and self.reset()  # Reset agent parameters if success/collision
            self.sensor.radar.observation(vehicles)  # Get distance/velocity to neighbouring vehicles

        def behavioural_planner():
            # Behaviour decision-making
            # Update the agent's lane index
            if abs(self.y_pos - self.goal_pos[1]) < 4:
                self.lane = [lane for lane, pos in LANES.items() if pos == self.goal_pos[1]][0]

            # Behaviour tree ----
            self.bt_agent_data.initialise(self)  # Update agent data to send to the BT
            self.bt_send.put(self.bt_agent_data)  # Send latest agent data to the queue
            while not self.bt_return.empty():
                self.goal_pos = self.bt_return.get()  # Get new goal position from the BT

        def path_planner():
            cars = [car.rect for car in self.sensor.radar.closest_cars if car]  # List of pygame.rect objects
            # Send data for the path planner
            self.send_queue.put(((self.x_pos, self.y_pos, self.theta), (self.goal_pos[0], self.goal_pos[1], 0), cars))
            while not self.return_queue.empty():
                self.path = self.return_queue.get()  # Retrieve path from the path planner

        def motion_planner():
            # Motion planning/controllers---
            def adaptive_cruise_control():
                """Calculate required acceleration of the agent
                        increment agent.velocity_x
                        Keep controller action and max velocity bounded"""
                if self.sensor.radar.closest_cars[0]:
                    self.acc = self.speed_controller.speed_control.send([self.velocity, self.sensor.radar.distances[0],
                                                                    self.sensor.radar.closest_cars[
                                                                        0].x_velocity]) * 2.237  # Update agent velocity in mph/s
                else:
                    self.acc = self.speed_controller.speed_control.send(
                        [self.velocity, 1000, self.max_speed]) * 2.237  # Get new acc in mph/s

            def lateral_control():
                if self.path:
                    try:
                        (self.traj_y_pos, self.traj_heading) = self.path[ceil(self.x_pos)]  # Update agent y-position
                    except KeyError:
                        pass
                    finally:
                        self.path = {k: v for k, v in self.path.items() if k >= self.x_pos}

                def crosstrack_error():
                    X_xte = self.x_pos
                    Y_xte = self.traj_y_pos
                    xte = sqrt((self.x_pos - X_xte) ** 2 + (self.y_pos - Y_xte) ** 2)  # The magnitude of the cross-track error

                    # Determine the offset direction
                    xte_vec = np.array([(X_xte - self.x_pos), (Y_xte - self.y_pos)])
                    heading = np.array([self.dr * cos(self.theta), - self.dr * sin(self.theta)])
                    Z = -np.cross(heading, xte_vec)
                    self.xte = np.sign(Z) * abs(xte)

                def heading_error():
                    if self.traj_heading > pi:
                        self.traj_heading = self.traj_heading - 2*pi

                    self.phi = self.theta - self.traj_heading

                    if self.phi > pi:
                        self.phi = self.phi - 2*pi

                def calculate_steering_angle():
                    # Calculate gain matrix using optimal control
                    k, _, _ = self.lateral_controller.trajectory_control.send(self.velocity / 2.237)
                    delta = -k * [self.xte, self.phi] * [1, 1]  # Controller action is u = -Kx
                    self.delta = np.sum(delta)
                    max_delta = 24 * 2 * pi / 360
                    if self.delta > max_delta:
                        self.delta = max_delta
                    elif self.delta < -max_delta:
                        self.delta = -max_delta

                    return

                crosstrack_error()  # Calculate the current cross-track error
                heading_error()  # Calculate the current heading error
                calculate_steering_angle()  # Calculate the required steering angle

            adaptive_cruise_control()
            lateral_control()

        def update_vehicle_state():
            def update_heading_angle():
                # Yaw rate = Velocity / Radius
                # Steering angle = length / Radius
                # => Radius = length / steering angle
                try:
                    if self.delta == 0:
                        yaw_rate = 0
                    else:
                        radius = self.length / self.delta
                        yaw_rate = (self.velocity / 2.237) / radius
                except ZeroDivisionError:
                    yaw_rate = 0
                finally:
                    self.theta += yaw_rate * 1 / scene.FPS
                    self.theta %= (2 * pi)

                return

            self.velocity += self.acc * (1 / FPS)

            self.dr = (self.velocity * 1 / FPS) + 0.5 * self.acc * (1 / FPS) ** 2
            self.r += self.dr  # Increment agent position

            self.x_pos += self.dr * cos(self.theta)
            self.y_pos -= self.dr * sin(self.theta)  # Pygame y-axis points downwards
            update_heading_angle()  # Calculate the new vehicle direction

            self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the agent center

        perception()
        behavioural_planner()
        path_planner()
        motion_planner()
        update_vehicle_state()

    def change_lane(self, direction):
        """Change the lane of the agent car."""
        if self.y_pos != self.goal_pos[1]:
            if direction == "l":
                if self.lane > 3:
                    self.lane -= 1
            elif direction == "r":
                if self.lane < 5:
                    self.lane += 1

    @staticmethod
    def get_path(path_planner, send_queue, return_queue):
        """
        Get inputs from send queue
        Run hybrid A* algorithm
        Put generated path in return queue
        wait for new inputs
        """
        # hybrid = HybridAstar()
        search_algorithm = path_planner()
        logger.debug(f'Initialised the {path_planner.__name__} path planner for the agent.')
        while True:
            if not send_queue.empty():
                while not send_queue.empty():
                    start_pos, goal_pos, obstacles = send_queue.get()

                new_path = search_algorithm.run(start_pos, goal_pos, obstacles)
                if new_path:
                    path = new_path
                    try:
                        xs, ys, deltas = itp.splev(np.linspace(0, 1, abs(int(goal_pos[0] - start_pos[0] + 1))), path)
                    except ValueError:
                        print(f"Invalid input data provided for splev: Start pos: {start_pos}, Goal pos: {goal_pos}")
                    # return_queue.put(dict(zip(xs.astype(int), ys)))
                    return_queue.put(dict(zip(xs.astype(int), zip(ys, deltas))))
                    # print(xs.astype(int))


class Sensor:
    def __init__(self, agent):
        self.agent = agent
        self.collision = Collision(agent)
        self.radar = Radar(agent)


class Collision:
    def __init__(self, agent):
        self.agent = agent

    def observation(self, vehicles):
        # car_list = [vehicle.rect for vehicle in vehicles] if vehicles else []  # Get list of rect objects
        car_list = [vehicle.rect for lane in vehicles for vehicle in lane] if vehicles else []  # Get list of rects
        collide = self.agent.rect.collidelist(car_list)  # Check if the agent is colliding with any vehicles

        # If the car from the car list is colliding with the agent
        if collide != -1:
            self.agent.color = (255, 0, 0)  # change the color to red
            if self.agent.x_pos > scene.DEAD_ZONE:
                pygame.image.save(world.display_surface, f"CollisionHistory/{time.time()}.jpeg")
                self.agent.n_hits += 1  # Do not penalise collisions where a npc spawns on top of the agent.
                self.agent.terminal_count += 1
            return True  # Finish the frame
        elif self.agent.x_pos >= scene.WINDOW_WIDTH:  # Finish the frame the agent has reached the end of the road.
            self.agent.successes += 1
            self.agent.terminal_count += 1
            return True
        else:
            return False


class Radar:
    sensor_range = 250

    def __init__(self, agent):
        self.agent = agent
        self.closest_cars = [None for _ in range(6)]
        self.distances = [None for _ in range(6)]

    def observation(self, vehicles: [[Vehicle]]) -> [[Vehicle], [int]]:
        """x = lane - self.agent.lane
                f(-1) = 2, f(0) = 0, f(1) = 4
                f(x) = 3 * x**2 + x

                Need to ignore other x values
                """
        car_lists = [[], [], [], [], [], []]  # Nested list to store vehicles by lane index

        def index(x):
            return 3 * x ** 2 + x

        def cond(v):
            return index(v.lane - self.agent.lane) if v.x_pos > self.agent.x_pos else index(
                v.lane - self.agent.lane) + 1

        # vehicles and [car_lists[cond(v)].append(v) for v in vehicles if abs(v.lane - self.agent.lane) <= 1]
        vehicles and [car_lists[cond(v)].append(v) for lane in vehicles for v in lane if
                      lane and abs(v.lane - self.agent.lane) <= 1]

        # Get the closest vehicle in each lane
        """Iterate through each lane in car_lists
            Calculate the relative distance to each vehicle
            keep the closest vehicle
            return list containing closest vehicle in each lane"""
        self.closest_cars = [[] for _ in range(6)]
        self.distances = [self.sensor_range for _ in range(6)]

        def rel_distance(v: Vehicle) -> int:
            return min([abs(self.agent.rect.left - v.rect.right), abs(v.rect.left - self.agent.rect.right)])

        # self.closest_cars = [min(lane_direction, key=rel_distance, default=None) for lane_direction in car_lists]
        # self.distances = [rel_distance(car) if car is not None else self.sensor_range for car in self.closest_cars]

        for i, lane_direction in enumerate(car_lists):
            for vehicle in lane_direction:
                if rel_distance(vehicle) < self.distances[i] and rel_distance(vehicle) != self.agent.rect.width:
                    self.distances[i] = rel_distance(vehicle)
                    self.closest_cars[i] = vehicle

        return self.closest_cars, self.distances

    def draw(self, surface):
        pass


# =================================
# Functions
# =================================

def main():
    world = World()  # Create the Pygame simulation environment
    world.TM = TrafficManager()  # Add traffic to the simulation
    world.agent = Agent(path_planner=HybridAstar)  # Add the agent vehicle to the simulation
    # Live graph
    try:
        world.game_loop()
    finally:
        pygame.quit()


if __name__ == "__main__":
    world = World()  # Create the Pygame simulation environment
    world.TM = TrafficManager()  # Add traffic to the simulation
    world.agent = Agent(path_planner=HybridAstar)  # Add the agent vehicle to the simulation
    try:
        world.game_loop(live_data=True)
    except Exception as e:
        logger.warning(f"Encountered e")
    finally:
        pygame.quit()
        sys.exit()
