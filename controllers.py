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
        front_car_velocity = self.sensor.closest_cars[0].x_velocity if self.sensor.closest_cars[0] else self.agent.max_speed
        # if self.sensor.closest_cars[0]:
        #     front_car_velocity = self.sensor.closest_cars[0].x_velocity
        #     else:
        #         self.agent.max_speed

        s_safe = 2 * front_car_velocity / 2.237  # Following 2-second rule. Velocities in m/s
        s_prime = max(s_safe, 3)  # The desired distance to the front vehicle (defaults to 3m if agent is stationary).

        acc1 = self.distance_control.send((s_prime, self.sensor.distances[0]))  # PV and SP swapped to ensure correct sign.
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
    def __init__(self, agent):
        self.agent = agent
        self.sensor = agent.sensor.radar
        self.minimum_spacing = 3
        self.time_headway = 1
        self.max_acceleration = self.agent.acceleration  # Max acceleration of the agent
        self.max_velocity = self.agent.max_speed  # Speed agent would drive at without traffic
        self.delta = 4  # Acceleration exponent
        self.speed_control = self.control()
        self.speed_control.send(None)

    def control(self):
        acc = None
        while True:
            yield acc
            s_star = self.minimum_spacing + self.agent.x_velocity * self.time_headway
            # If the desired follow distance is greater than the gap between front/behind vehicle
            if s_star > self.sensor.distances[0] + self.sensor.distances[1]:
                s_star = 0.5 * (self.sensor.distances[0] + self.sensor.distances[1])
            try:
                acc = self.max_acceleration * (1 - (self.agent.x_velocity / self.max_velocity) ** self.delta - (s_star / self.sensor.distances[0]) ** 2)
            except ZeroDivisionError:
                logger.debug(f'{self.__class__.__name__}: IDM: Zero division attempted. Looks like there was an accident')


def idm(desired_velocity, max_acceleration, acc_bar=0):
    desired_velocity = desired_velocity  # The velocity the vehicle would drive at in free traffic
    minimum_spacing = 3  # The minimum desired net distance. (Car can't move if distance to the car in front isn't greater than this.)
    time_headway = 1  # The minimum possible time to the vehicle in front
    max_acc = max_acceleration  # The max vehicle acceleration
    delta = 4  # Acceleration exponent
    acc = acc_bar # Initial control
    while True:
        velocity_x, front_car_distance, front_car_velocity = yield acc
        s_star = minimum_spacing + velocity_x * time_headway
        acc = max_acc * (1 - (velocity_x / desired_velocity) ** delta - (s_star / front_car_distance) ** 2)
