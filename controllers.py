from math import sqrt

from scenario.highway import FPS

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler('controllers.log' if __name__ == "__main__" else f'LogFiles/{__name__}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger


class PID:
    def __init__(self, agent):
        self.agent = agent
        self.sensor = agent.sensor.radar
        self.distance_control = self.control(0.00001, 0.01, 0.0005)
        self.velocity_control = self.control(0.00001, 0.01, 0.0005)
        self.distance_control.send(None), self.velocity_control.send(None)  # Initialise generator functions

    def send(self):
        front_car_velocity = self.sensor.closest_cars[0].x_velocity if self.sensor.closest_cars[
            0] else self.agent.max_speed
        # if self.sensor.closest_cars[0]:
        #     front_car_velocity = self.sensor.closest_cars[0].x_velocity
        #     else:
        #         self.agent.max_speed

        s_safe = 2 * front_car_velocity / 2.237  # Following 2-second rule. Velocities in m/s
        s_prime = max(s_safe, 3)  # The desired distance to the front vehicle (defaults to 3m if agent is stationary).

        acc1 = self.distance_control.send(
            (s_prime, self.sensor.distances[0]))  # PV and SP swapped to ensure correct sign.
        acc2 = self.velocity_control.send((self.agent.x_velocity / 2.237, front_car_velocity / 2.237))  # Inputs in m/s
        acc = acc1 + acc2  # Net control action is sum of distance and velocity controller actions.
        if acc > self.agent.acceleration:
            acc = self.agent.acceleration
        elif acc < -self.agent.deceleration:
            acc = -self.agent.deceleration
        return acc

    @staticmethod
    def control(Kp, Ki, Kd, MV_bar=0):
        # Initialise stored data
        e_prev = 0  # Previous error value
        I = 0  # Integral value
        MV = MV_bar  # Initial control
        while True:
            PV, SP = yield MV  # yield MV, wait for new PV, SP

            # PID calculations
            e = SP - PV  # Tracking error
            P = Kp * e  # Proportional term
            I = I + Ki * e / FPS  # Integral term
            D = Kd * (e - e_prev) * FPS  # Derivative term

            MV = MV_bar + P + I + D  # Controller action is sum of 3 terms
            e_prev = e  # Update stored data for next iteration


class IDM:
    """
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
            desired_acceleration = (1 - (velocity_x / self.desired_velocity) ** self.delta)

            try:
                braking_term = (s_star / front_car_distance) ** 2
            except ZeroDivisionError:
                braking_term = 0
                logger.debug("Encountered ZeroDivisionError - looks like a car crashed.")

            acc = self.style['acceleration'] * (desired_acceleration - braking_term)
