# =================================
# Imports
# =================================
import os
import pygame
import time
import random
import threading

from behaviour_executive import finite_state_machine as fsm
from behaviour_executive import behaviour_tree as bt
from path_planner.hybrid_astar import HybridAstar
from path_planner.normal_astar import Astar
from multiprocessing import Process, Queue
from scipy import interpolate as itp
import numpy as np
from math import ceil

from controllers import PID, IDM

from scenario.highway import LANES, FPS, LANE_VELOCITIES
import scenario.highway as scene

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler(f'LogFiles/{__name__}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger

# =================================
# Global variables
# =================================
TIME_GAIN = 1  # Speed up the simulation # todo Doesn't affect path planner or BT subprocesses
FPS /= TIME_GAIN

TRAFFIC_DENSITY = 1  # Density of NPC vehicles in the simulation

# =============================
# Classes
# =============================


class World:
    """A class that handles the pygame simulation environment"""

    def __init__(self):
        self.display_surface = World.init_pygame()  # Open a new named Pygame window
        image = pygame.image.load(r'highway.jpg').convert()
        self.image = pygame.transform.scale(image,
                                            (scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT))  # Scale the background image
        # image = pygame.image.load(r'Roundabout2.jpg').convert()
        # self.image = pygame.transform.scale(image,
        #                                     (scene.WINDOW_WIDTH, scene.WINDOW_HEIGHT))  # Scale the background image

        self.display_surface.blit(self.image, (0, 0))  # Fill the pygame window with highway image

        self.TM = None  # Create a new instance of the traffic manager
        self.agent = None  # Create the agent
        self.start_time = 0  # Stores the start time of the simulation
        logger.info(f'Created new instance of the simulation environment.')

    def game_loop(self) -> None:
        self.start_time = time.time()  # Store the start time of the simulation
        clock = pygame.time.Clock()  # Controls the frame rate
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

                elapsed_time = time.time() - self.start_time  # Calculate elapsed time

                self.TM and self.update_vehicles()  # Update NPC vehicles
                self.agent and self.update_agent()  # Update and draw the agent

                self.draw_window()  # Draw objects on the pygame window
                pygame.display.update()  # Update the pygame window
                self.caption(elapsed_time, clock)  # Update the window caption

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

    def caption(self, elapsed_time: float, clock: pygame.time.Clock) -> None:
        tracks_count = self.TM.lane_distribution() if self.TM else 0
        cap = []
        sim_cap = f"Elapsed time: {round(elapsed_time)}, FPS: {round(clock.get_fps())}"
        cap.append(sim_cap)
        if self.TM:
            tm_cap = f"Num Vehicles: {self.TM and len(self.TM.vehicles)}, Lane dist: {tracks_count}"
            cap.append(tm_cap)
        if self.agent:
            # todo Caption should only show current state if FSM is active
            agent_cap = f"Scenarios simulated: {self.agent.terminal_count}, Collisions: {self.agent.n_hits}, " \
                        f"Speed: {round(self.agent.x_velocity)} mph, " \
                        f"Goal pos: {[round(v) for v in self.agent.goal_pos], [round(v) for v in self.agent.goal_pos2]}, " \
                        f"Current state: {str(self.agent.fsm.state)}, Lane: {self.agent.lane}"
            cap.append(agent_cap)

        caption = ", ".join(cap)
        pygame.display.set_caption(caption)

    def update_vehicles(self) -> None:
        """Updates the positions of all traffic on the screen. Removes vehicles that have moved off-screen."""
        [car.update_position() if 0 <= car.x_pos <= scene.WINDOW_WIDTH else self.TM.vehicles.remove(car)
         for car in self.TM.vehicles]

    def update_agent(self) -> None:
        self.agent.update_position(self.TM and self.TM.vehicles)  # Update the agent

    def draw_window(self) -> None:
        def draw_radar(horizontal: bool = False) -> None:
            [pygame.draw.line(self.display_surface, (255, 0, 0),
                              (self.agent.x_pos, LANES[car.lane] if horizontal else self.agent.y_pos),
                              (car.x_pos, LANES[car.lane]))
             for car in self.agent.sensor.radar.closest_cars if car and car.lane > 2]

        self.display_surface.blit(self.image, (0, 0))  # Fill the pygame window with the background image
        self.agent and draw_radar(horizontal=True)
        # Draw traffic
        self.TM and [pygame.draw.rect(self.display_surface, car.colour, car.rect) for car in self.TM.vehicles]
        if self.agent:
            # Draw path
            (len(self.agent.path) > 1 and pygame.draw.lines(self.display_surface, (250, 0, 250),
                                                            False, list(self.agent.path.items())))
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
        self.timed_spawn = RepeatedTimer(1 / (TRAFFIC_DENSITY * TIME_GAIN), self.spawn_vehicles)
        logger.info('Created new instance of the traffic manager.')

    def spawn_vehicles(self):
        """Creates a new vehicle and appends it to the list of vehicles."""
        max_vehicles = 3
        new_vehicles = []  # Temporary list to store our new vehicles.
        number_of_vehicles = random.randint(0, max_vehicles)  # Assign a random number of vehicles to spawn in a range.

        for i in range(number_of_vehicles):
            new_vehicle = Vehicle()  # Create a new vehicle
            for Npc in range(len(new_vehicles)):  # Iterate through the list of new vehicles
                # If multiple vehicles being added they must be in different lanes
                # and not collide with existing vehicles.
                if new_vehicle.lane == new_vehicles[Npc].lane:
                    break
            else:
                for car in self.vehicles:
                    if car.lane == new_vehicle.lane and (
                            car.rect.x <= new_vehicle.rect.x <= car.rect.x + car.rect.width) or (
                            car.rect.x <= new_vehicle.rect.x + new_vehicle.rect.width <= car.rect.x + car.rect.width):
                        break
                else:
                    new_vehicles.append(new_vehicle)  # Add the vehicle to the list of new vehicles

        self.vehicles.extend(new_vehicles)  # Add the new vehicles to the vehicle list

    def lane_distribution(self):
        """Returns the number of vehicles in each lane."""
        tracks = [vehicle.lane for vehicle in self.vehicles]
        tracks_count = [tracks.count(track) for track in range(6)]

        return tracks_count


class Vehicle:
    """Car class for NPC vehicles"""
    scale = 10 / 3

    def __init__(self):
        # The probability of a vehicle spawning in the top three lanes.
        # Accounts for bias of random.choice towards lanes 0,1,2 with higher values of traffic density
        self.lane_bias = 0.04 if TRAFFIC_DENSITY > 25 else -0.008 * TRAFFIC_DENSITY + 0.24

        # Geometric parameters and initial lane
        self.width = random.normalvariate(2, 0.15) * self.scale  # Width of the vehicle (y-direction)
        if self.width / self.scale > 2.15:  # For spawning trucks
            self.length = random.normalvariate(16.0, 2) * self.scale  # For trucks
            # Trucks won't spawn in the fast lane
            self.lane = random.choice([4, 5]) if random.random() > self.lane_bias else random.choice([0, 1])
        else:
            self.length = random.normalvariate(4.5, 0.2) * self.scale  # Length of the vehicle (x-direction)
            # Choose a random lane for the vehicle
            self.lane = random.choice([3, 4, 5]) if random.random() > self.lane_bias else random.choice([0, 1, 2])

        # Kinematic parameters
        self.acceleration = 0  # The initial acceleration of the vehicle
        self.x_velocity = LANE_VELOCITIES[self.lane]  # x velocity of the vehicle
        self.y_velocity = 0

        # Set spawn location
        self.y_pos = LANES[self.lane]  # The y-position of the vehicle's lane
        self.x_pos = 0 if self.lane > 2 else scene.WINDOW_WIDTH  # Set the start x position depending on travel direction

        # Assign the vehicle a random colour
        self.colour = pygame.Color(int(random.random() * 255), int(random.random() * 255), int(random.random() * 255))

        self.rect = pygame.Rect(0, 0, self.length, self.width)  # Create the vehicle's rectangle object
        self.rect.center = (self.x_pos, self.y_pos)

    def update_position(self):
        # increment position values: s = ut + 0.5at^2
        self.x_pos += self.x_velocity * 1 / FPS
        self.y_pos += self.y_velocity * 1 / FPS
        self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the vehicle rectangle


class RepeatedTimer:
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        # self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class Agent:
    """Car class for the Agent vehicle"""
    length = 15  # length (x-dim) of the car
    width = 8  # width (y-dim) of the car

    min_speed = 30  # Min speed of the agent car (mph)
    max_speed = 1.1 * scene.SPEED_LIMIT  # Max speed (mph)
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

        self.pid = PID(self)  # Initialise PID controller
        self.speed_controller = speed_controller(self)  # Initialise speed controller

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
            self.goal_pos2 = self.goal_pos
            self.x_velocity = agent.x_velocity
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
        self.x_pos = 0
        self.y_pos = LANES[self.lane]
        self.x_velocity = LANE_VELOCITIES[self.lane]
        self.goal_pos = [self.x_pos + 40, self.y_pos]  # Goal position for the path planner
        self.goal_pos2 = self.goal_pos
        self.path = {}

    def update_position(self, vehicles):
        self.sensor.collision.observation(vehicles) and self.reset()  # Reset agent parameters if success/collision
        self.sensor.radar.observation(vehicles)  # Get distance/velocity to neighbouring vehicles

        # Motion planning/controllers---
        """Calculate required acceleration of the agent
                increment agent.velocity_x
                Keep controller action and max velocity bounded"""
        # self.x_velocity += self.pid.send() * 2.237  # Update agent velocity in mph/s
        self.x_velocity += self.speed_controller.speed_control.send(None) * 2.237  # Update agent velocity in mph/s

        # Ensure agent velocity is within acceptable range
        if self.max_speed <= self.x_velocity:
            self.x_velocity, acc = self.max_speed, 0
        elif self.min_speed >= self.x_velocity:
            self.x_velocity, acc = self.min_speed, 0

        # Behaviour decision-making # todo Easily switch between FSM and BT
        # Increment the agent.lane when we change to a new lane
        if abs(self.y_pos - self.goal_pos[1]) < 2:
            self.lane = [lane for lane, pos in LANES.items() if pos == self.goal_pos[1]][0]
        # Finite state machine
        # self.goal_pos = self.fsm.on_event()  # Update behavioural planner # todo Fix the finite state machine

        # Behaviour tree ----
        """Send the current agent instance to the queue on each iteration
                receive new goal position from return queue"""
        self.bt_agent_data.initialise(self)  # Update agent data to send to the BT
        self.bt_send.put(self.bt_agent_data)  # Send latest agent data to the queue
        while not self.bt_return.empty():
            # _ = self.bt_return.get()  # Must empty the queue into a throwaway variable so while loop finishes
            self.goal_pos = self.bt_return.get()  # Get new goal position from the BT
        # ---

        cars = [car.rect for car in self.sensor.radar.closest_cars if car]  # List of pygame.rect objects
        # Send data for the path planner
        # todo Consider modifying the start position of the next path search to be the goal position of the previous
        #  if the action hasn't changed.
        self.send_queue.put(((self.x_pos, self.y_pos, 0), (self.goal_pos[0], self.goal_pos[1], 0), cars))
        # Retrieve path from the path planner
        while not self.return_queue.empty():
            self.path = self.return_queue.get()

        self.x_pos += self.x_velocity * 1 / FPS  # Increment agent position
        if self.path:
            try:
                self.y_pos = self.path[ceil(self.x_pos)]  # Update agent y-position
            except KeyError:
                # self.x_pos can take a value >WINDOW_WIDTH when the agent reaches edge of the screen
                pass

        self.path = {k: v for k, v in self.path.items() if k >= self.x_pos}
        self.rect.center = (self.x_pos, self.y_pos)  # Update the location of the agent center
        """Decision module
        FSM
        Observe surrounding vehicles
        Transition states
        Execute appropriate action
        """
        """list of rect objects and list of velocities for all vehicles in simulation
        Check if colliding with anything (Collision sensor)
        
        Radar detection
        
        Decision module -> returns goal position
        
        Path planner (A* / Hybrid A*)
        
        Motion planner (controllers PID/IDM)
        
        
        """

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
                        xs, ys = itp.splev(np.linspace(0, 1, abs(int(goal_pos[0] - start_pos[0] + 1))), path)
                    except ValueError:
                        print(f"Invalid input data provided for splev: Start pos: {start_pos}, Goal pos: {goal_pos}")
                    return_queue.put(dict(zip(xs.astype(int), ys)))
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
        car_list = [vehicle.rect for vehicle in vehicles] if vehicles else []  # Get list of rect objects
        collide = self.agent.rect.collidelist(car_list)  # Check if the agent is colliding with any vehicles

        # If the car from the car list is colliding with the agent
        if collide != -1:
            self.agent.color = (255, 0, 0)  # change the color to red
            if self.agent.x_pos > scene.DEAD_ZONE:
                self.agent.n_hits += 1  # Do not penalise collisions where a npc spawns on top of the agent.
                self.agent.terminal_count += 1
            return True  # Finish the frame
        elif self.agent.x_pos >= scene.WINDOW_WIDTH:  # Finish the frame the agent has reached the end of the road.
            self.agent.successes += 1
            self.agent.terminal_count += 1
            return True
        else:
            # self.agent.color = (0, 0, 0)
            return False


class Radar:
    sensor_range = 250

    def __init__(self, agent):
        self.agent = agent
        self.closest_cars = [None for _ in range(6)]
        self.distances = [None for _ in range(6)]

    def observation(self, vehicles):
        """x = lane - self.agent.lane
                f(-1) = 2, f(0) = 0, f(1) = 4
                f(x) = 3 * x**2 + x

                Need to ignore other x values
                """
        car_lists = [[], [], [], [], [], []]  # Nested list to store vehicles by lane index

        # index = lambda x: 3 * x ** 2 + x

        def index(x):
            return 3 * x ** 2 + x

        def cond(v):
            return index(v.lane - self.agent.lane) if v.x_pos > self.agent.x_pos else index(
                v.lane - self.agent.lane) + 1

        vehicles and [car_lists[cond(v)].append(v) for v in vehicles if abs(v.lane - self.agent.lane) <= 1]

        # Get the closest vehicle in each lane
        """Iterate through each lane in car_lists
            Calculate the relative distance to each vehicle
            keep the closest vehicle
            return list containing closest vehicle in each lane
                """
        self.closest_cars = [[] for _ in range(6)]
        self.distances = [self.sensor_range for _ in range(6)]

        # rel_distance = lambda v: self.agent.rect.left - v.rect.right if i % 2 else v.rect.left - self.agent.rect.right

        def rel_distance(v):
            return self.agent.rect.left - v.rect.right if i % 2 else v.rect.left - self.agent.rect.right

        for i, lane_direction in enumerate(car_lists):
            for vehicle in lane_direction:
                if rel_distance(vehicle) < self.distances[i]:
                    self.distances[i] = rel_distance(vehicle)
                    self.closest_cars[i] = vehicle

        return self.closest_cars, self.distances

    def draw(self, surface):
        pass


if __name__ == "__main__":
    world = World()  # Create the Pygame simulation environment
    world.TM = TrafficManager()  # Add traffic to the simulation
    world.agent = Agent(path_planner=HybridAstar)  # Add the agent vehicle to the simulation
    # Live graph
    try:
        world.game_loop()
    finally:
        pygame.quit()
