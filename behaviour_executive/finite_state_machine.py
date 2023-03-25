from scenario.highway import LANES

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('finite_state_machine.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger


class State(object):
    """Defines a state object which provides some utility functions for the individual states within the state machine.
    """

    def __init__(self, agent, sensor):
        self.agent = agent
        self.sensor = sensor
        self.observation = None

    def on_event(self):
        """Handle events that are delegated to this state."""
        pass

    def __repr__(self):
        """Leverages the __str__ method to describe the state."""
        return self.__str__()

    def __str__(self):
        """Returns the name of the state."""
        return self.__class__.__name__


class FreeRideState(State):
    """The state which indicates that there are no other vehicles in front of the AV."""

    def on_event(self):
        if self.observation == 'right_lane_free':
            if self.agent.lane == 5:
                pass
            else:
                return LaneChangeRightState(self.agent, self.sensor)
        elif self.observation == 'clear_road':
            pass
        elif self.observation == 'vehicle_ahead' or self.observation == 'slow_vehicle':
            return FollowState(self.agent, self.sensor)

        return self

        # if self.vehicle_ahead() or self.slow_vehicle():
        #     return FollowState(self.agent, self.sensor)
        # if self.right_lane_free() and self.agent.lane != 5:
        #     return LaneChangeRightState(self.agent, self.sensor)
        # return self

    def action(self):
        return [self.agent.x_pos + 0.5 * self.sensor.radar.distances[0], LANES[self.agent.lane]]


class FollowState(State):
    """The state which indicates that the AV is following a lead vehicle."""

    def on_event(self):
        if self.observation == 'clear_road':
            return FreeRideState(self.agent, self.sensor)
        elif self.observation == 'slow_vehicle':
            if self.agent.lane == 3:
                pass
            else:
                return LaneChangeLeftState(self.agent, self.sensor)
            # elif observation == 'left_lane_free':
            #     return LaneChangeLeftState(self.agent, self.sensor)
        elif self.observation == 'right_lane_free':
            return LaneChangeRightState(self.agent, self.sensor)

        return self

        # if self.slow_vehicle() and self.agent.lane != 3:
        #     return LaneChangeLeftState(self.agent, self.sensor)
        # elif self.clear_road():
        #     return FreeRideState(self.agent, self.sensor)
        # return self

    def action(self):
        return [self.agent.x_pos + 0.5 * self.sensor.radar.distances[0], LANES[self.agent.lane]]


class LaneChangeLeftState(State):
    """The state which indicates a left lane change."""
    def left_lane_free(self):
        """Left lane free only if front vehicle is at least 2/3 the current follow distance in front,
            the separation between vehicles in the left lane is at least the safe follow distance and
            the distance to the vehicle behind in the left lane is at least the safe follow distance."""
        if self.sensor.radar.closest_cars[2]:
            return (self.sensor.radar.distances[2] > self.sensor.radar.distances[0] * 2 / 3) and \
                   (self.sensor.radar.distances[2] + self.sensor.radar.distances[3] > 2 *
                    self.sensor.radar.closest_cars[2].x_velocity / 2.237 + 2 * self.agent.length) and \
                   (self.sensor.radar.distances[3] > 2 * self.sensor.radar.closest_cars[2].x_velocity / 2.237)
        else:
            return True

    def on_event(self):
        if self.observation == 'slow_vehicle':
            if self.agent.lane == 3:
                return FollowState(self.agent, self.sensor)
        elif self.observation == 'clear_road':
            return FreeRideState(self.agent, self.sensor)
        elif self.observation == 'vehicle_ahead' or self.observation == 'slow_vehicle':
            return FollowState(self.agent, self.sensor)

        return self

        # if self.clear_road():
        #     return FreeRideState(self.agent, self.sensor)
        # elif self.slow_vehicle() and self.agent.lane == 3 or self.vehicle_ahead():
        #     return FollowState(self.agent, self.sensor)
        # return self

    def action(self):
        if self.left_lane_free():
            goal_pos = [self.agent.x_pos + 0.5 * self.sensor.radar.distances[2], LANES[self.agent.lane - 1]]
            # self.agent.change_lane('l')
            return goal_pos
        else:
            return self.agent.goal_pos


class LaneChangeRightState(State):
    """The state which indicates a right lane change."""

    def on_event(self):
        if self.observation == 'right_lane_free':
            if self.agent.lane == 5:
                return FreeRideState(self.agent, self.sensor)
        elif self.observation == 'vehicle_ahead' or self.observation == 'slow_vehicle':
            return FollowState(self.agent, self.sensor)
        elif self.observation == 'clear_road':
            return FreeRideState(self.agent, self.sensor)

        return self

        # if self.vehicle_ahead():
        #     return FollowState(self.agent, self.sensor)
        # return FreeRideState(self.agent, self.sensor)

    def action(self):
        goal_pos = [self.sensor.radar.distances[4] + self.agent.x_pos, LANES[self.agent.lane + 1]] \
            if self.agent.lane < 5 else [self.agent.x_pos + 0.5 * self.sensor.radar.distances[0], self.agent.y_pos]
        # self.agent.change_lane('r')
        return goal_pos


class HighwayDrive:
    """A simple state machine that mimics basic highway driving behaviours."""

    def __init__(self, agent, sensor):
        """ Initialise the components. """
        self.state = FreeRideState(agent, sensor)  # Default state
        self.agent = agent
        self.sensor = sensor

    # ----
    # Transition logic
    # ----
    def slow_vehicle(self):
        return (self.sensor.radar.distances[0] < 100) and (
                self.sensor.radar.closest_cars[0].x_velocity < self.agent.max_speed - 10)

    def vehicle_ahead(self):
        return self.sensor.radar.distances[0] < 100

    def right_lane_free(self):
        return (self.sensor.radar.distances[4] > 200) and (self.sensor.radar.distances[5] > 20)

    def clear_road(self):
        return self.sensor.radar.distances[0] >= 100

    def observe_surrounding_vehicles(self):
        if self.slow_vehicle():
            return 'slow_vehicle'
        elif self.vehicle_ahead():
            return 'vehicle_ahead'
        elif self.right_lane_free():
            return 'right_lane_free'
        elif self.clear_road():
            return 'clear_road'

    def on_event(self):
        """Incoming events are delegated to the given state which then handles the event.
        The result is assigned as the new state."""
        if abs(self.agent.y_pos - self.agent.goal_pos[1]) < 2:
            self.state.observation = self.observe_surrounding_vehicles()
            logger.debug(f'{self.state} -> {self.state.observation} -> {self.state.on_event()}')
            self.state = self.state.on_event()  # The next state will be the result of the on_event function
            # logger.debug(f'Returned new state: {self.state}')

            return self.state.action()
        else:
            return self.agent.goal_pos
