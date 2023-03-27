"""A module containing helper functions for the Action behaviours in the behaviour tree."""
from scenario.highway import LANES


def free_ride(agent: isinstance) -> tuple[float, float]:
    goal_pos = (agent.x_pos + 0.5 * agent.front_car_distance, LANES[agent.lane])
    return goal_pos


def follow(agent: isinstance) -> tuple[float, float]:
    goal_pos = (agent.x_pos + 0.5 * agent.front_car_distance, LANES[agent.lane])
    return goal_pos


def lane_change_right(agent: isinstance) -> tuple[float, float]:
    goal_pos = (agent.x_pos + 0.5 * agent.front_right_car_distance, LANES[agent.lane + 1]) \
        if agent.lane < 5 else (agent.x_pos + 0.5 * agent.front_car_distance, LANES[agent.lane])
    return goal_pos


def lane_change_left(agent: isinstance) -> tuple[float, float]:
    goal_pos = (agent.x_pos + 0.5 * agent.front_left_car_distance, LANES[agent.lane - 1])
    return goal_pos
