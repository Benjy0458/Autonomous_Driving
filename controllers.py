from math import sqrt
from control import lqr
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler('controllers.log' if __name__ == "__main__" else f'LogFiles/{__name__}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger


class IDM:
    """A class that implements the Intelligent Driver Model for automated cruise control of a vehicle.
    time_headway: Desired time to front vehicle
    acceleration: Max acceleration
    deceleration: comfortable deceleration
    """
    driving_styles = {
        "safe": {
            # 'desired_velocity': desired_velocity,  # lane velocity
            'time_headway': 1.5,  # 0.8 - 2 s
            'acceleration': 2,  # 0.8 - 2.5 m/s^2
            'deceleration': 3,  # ~2 m/s^2
        },
        "aggressive": {
            # 'desired_velocity': desired_velocity,  # lane velocity
            'time_headway': 1,  # 0.8 - 2 s
            'acceleration': 3,  # 0.8 - 2.5 m/s^2
            'deceleration': 4,  # ~2 m/s^2
        },
        "wide_load": {
            # 'desired_velocity': desired_velocity,  # lane velocity
            'time_headway': 1.7,  # 0.8 - 2 s
            'acceleration': 1.5,  # 0.8 - 2.5 m/s^2
            'deceleration': 2,  # ~2 m/s^2
        },
    }
    delta = 4  # Acceleration exponent
    minimum_spacing = 3  # Zero velocity follow distance

    def __init__(self, driving_style: str, desired_velocity: float):
        self.style = self.driving_styles[driving_style]
        self.desired_velocity = desired_velocity
        self.sqrtab = sqrt(self.style['acceleration'] * self.style['deceleration'])
        self.speed_control = self.control()
        self.speed_control.send(None)

    def control(self):
        acc = 0
        while True:
            velocity_x, front_car_distance, front_car_velocity = yield acc

            # Desired dynamical distance
            s_star = self.minimum_spacing + max(0, velocity_x * self.style['time_headway'] +
                                                (velocity_x * (velocity_x - front_car_velocity) / (2 * self.sqrtab)))
            # Desired acceleration on a free road
            try:
                desired_acceleration = (1 - (velocity_x / self.desired_velocity) ** self.delta)
            except OverflowError:
                desired_acceleration = 0
                logging.warning(f"Encountered OverflowError calculating: "
                                f"1 - ({velocity_x} / {self.desired_velocity}) ** {self.delta}")

            try:
                braking_term = (s_star / front_car_distance) ** 2
            except ZeroDivisionError:
                braking_term = 0
                logger.debug(f"Encountered ZeroDivisionError calculating: "
                             f"braking_term = ({s_star} / {front_car_distance}) ** 2")

            acc = self.style['acceleration'] * (desired_acceleration - braking_term)


class OptimalControl:
    """A class that implements the optimal control algorithm for vehicle trajectory tracking."""
    def __init__(self, length):
        self.length = length
        self.Q = np.eye(2)
        self.R = np.eye(1)
        self.trajectory_control = self.control()
        self.trajectory_control.send(None)

    def control(self):
        K, P, E = 0, 0, 0
        while True:
            V = yield K, P, E

            A = np.array([[0, V], [0, 0]])
            B = -np.array([[V], [V/self.length]])
            try:
                K, P, E = lqr(A, B, self.Q, self.R)
            except Exception as e:
                logger.warning(f"Encountered {e} trying to solve lqr problem.")
