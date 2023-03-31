"""A module containing helper functions for the condition behaviours in the behaviour tree."""

# todo Modify conditions to use time_headway


def slow_vehicle(agent: isinstance) -> bool:
    return agent.front_car_velocity < agent.max_speed - 5


def vehicle_ahead(agent: isinstance) -> bool:
    return agent.front_car_distance < 100


def right_lane_free(agent: isinstance) -> bool:
    return (agent.front_right_car_distance > 20) and (agent.rear_right_car_distance > 20)


def left_lane_free(agent: isinstance) -> bool:
    """Left lane free only if front vehicle is at least 2/3 the current follow distance in front,
        the separation between vehicles in the left lane is at least the safe follow distance and
        the distance to the vehicle behind in the left lane is at least the safe follow distance."""
    return (agent.lane > 3) and (agent.front_left_car_distance > agent.front_car_distance * 2 / 3) and (
            agent.front_left_car_distance + agent.rear_left_car_distance >
            2 * agent.front_left_car_velocity / 2.237 + 2 * agent.length) and (
            agent.rear_left_car_distance > 2 * agent.front_left_car_velocity / 2.237)


def clear_road(agent: isinstance) -> bool:
    return agent.front_car_distance >= 100


def vehicle_approaching(agent: isinstance) -> bool:
    """If velocity of car behind in current lane is greater than target velocity and distance <= some_distance."""
    return agent.rear_car_velocity > agent.x_velocity + 3 and agent.rear_car_distance < 50
