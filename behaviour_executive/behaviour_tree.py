import py_trees
from multiprocessing import Queue
import sys
from typing import Callable
import curses

from behaviour_executive import situational_context, mission_planner


class AgentDummy:
    """A dummy class to represent the agent object."""
    def __init__(self) -> None:
        """Initialise variables to arbitrary defaults."""
        self.goal_pos = [0, 0]
        self.x_velocity = 70
        self.max_speed = 77
        self.length = 15

        self.x_pos = 0
        self.lane = 5

        # Surrounding vehicles
        self.front_car_distance = 100
        self.front_car_velocity = 50

        self.front_right_car_distance = 250
        self.rear_right_car_distance = 250

        self.front_left_car_distance = 250
        self.front_left_car_velocity = 60
        self.rear_left_car_distance = 250

        self.rear_car_velocity = 50
        self.rear_car_distance = 250

    def __str__(self) -> str:
        return str(self.__dict__)


class Condition(py_trees.behaviour.Behaviour):
    """A generic behaviour that checks a condition."""
    def __init__(self, name: str = "Condition", condition: Callable = None) -> None:
        """Configure the name of the behaviour"""
        super(Condition, self).__init__(name)
        self.condition = condition
        self.blackboard = self.attach_blackboard_client(name=f"{name}")
        # Set read only permissions for the agent data
        self.blackboard.register_key(
            key="agent", access=py_trees.common.Access.READ
        )
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def setup(self) -> None:
        """No delayed initialisation required"""
        self.logger.debug("%s.setup()" % self.__class__.__name__)

    def initialise(self) -> None:
        self.logger.debug("%s.initialise()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        """Check the condition and decide on a new status"""
        if self.condition(self.blackboard.agent):
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example"""
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class Action(py_trees.behaviour.Behaviour):
    """A generic behaviour that requests and action."""
    def __init__(self, name: str, behaviour: callable):
        """Configure the name of the behaviour."""
        super(Action, self).__init__(name)
        self.behaviour = behaviour
        self.blackboard = self.attach_blackboard_client(name=f"{name}")
        # Set read only permissions for the agent data
        self.blackboard.register_key(key="agent", access=py_trees.common.Access.READ)
        # Set write permissions for the new goal position
        self.blackboard.register_key(key="new_goal_pos", access=py_trees.common.Access.WRITE)
        self.logger.debug(f"{self.__class__.__name__}.__init__()->{str(self.behaviour)}")

    def setup(self, **kwargs: int) -> None:
        """Kickstart the separate process this behaviour will work with.
        Process is usually already running - setup method is only used to verify it exists."""
        self.logger.debug("%s.setup()" % self.__class__.__name__)

    def initialise(self) -> None:
        """Reset a counter variable."""
        self.logger.debug("%s.initialise()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        """Increment the counter, monitor and decide new status."""
        new_status = py_trees.common.Status.RUNNING
        self.blackboard.new_goal_pos = self.behaviour(self.blackboard.agent)

        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example."""
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


# Composites
def create_selector(children: [py_trees.behaviour.Behaviour], name: str = "Selector") -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
        Returns the root behaviour"""
    selector = py_trees.composites.Selector(name=name, memory=False)
    selector.add_children([*children])
    return selector


def create_sequence(children: [py_trees.behaviour.Behaviour], name: str = "Sequence") -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
        Returns the root behaviour"""
    sequence = py_trees.composites.Sequence(name=name, memory=False)
    sequence.add_children([*children])
    return sequence


def highway_drive():
    """Returns the root behaviour for the highway_drive behaviour tree."""
    # Left sub-tree -----------------
    # Conditions
    clear_road = Condition(name="Clear Road", condition=situational_context.clear_road)
    vehicle_approaching = Condition(name="Vehicle Approaching", condition=situational_context.vehicle_approaching)
    right_lane_free = Condition(name="Right Lane Free", condition=situational_context.right_lane_free)

    # Decorators
    inverter_vehicle_approaching = py_trees.decorators.Inverter(name="Inverter", child=vehicle_approaching)
    inverter_right_lane_free = py_trees.decorators.Inverter(name="Inverter", child=right_lane_free)

    # Actions
    lane_change_right = Action(name="Lane Change Right", behaviour=mission_planner.lane_change_right)
    free_ride = Action(name="Free Ride", behaviour=mission_planner.free_ride)

    # Composites
    yield_to_vehicle = create_selector([inverter_vehicle_approaching, inverter_right_lane_free, lane_change_right], name="Yield")
    free_ride_seq = create_sequence([yield_to_vehicle, clear_road, free_ride], name="Cruise")
    # ---------

    # Right sub-tree ---------
    # Conditions
    slow_vehicle = Condition(name="Slow Vehicle", condition=situational_context.slow_vehicle)
    left_lane_free = Condition(name="Left Lane Free", condition=situational_context.left_lane_free)

    # Actions
    lane_change_left = Action(name="Lane Change Left", behaviour=mission_planner.lane_change_left)
    follow = Action(name="Follow", behaviour=mission_planner.follow)

    # Composites
    overtake = create_sequence([slow_vehicle, left_lane_free, lane_change_left], name="Overtake")
    follow_sel = create_selector([overtake, follow], name="Follow")
    # ---------

    # Create root
    root = create_selector([free_ride_seq, follow_sel], name="Root")
    return root


def setup(bt_send: Queue, bt_return: Queue, root: callable(py_trees.behaviour.Behaviour) = highway_drive,
          console: object = curses.initscr):
    """Sets up and runs the behaviour tree provided.
    Inputs:
        bt_send: A multiprocessing.Queue() object to receive data from the parent process
        bt_return: A multiprocessing.Queue() object to send data to the parent process
        root: A callable function that returns the root behaviour of the tree"""
    py_trees.logging.level = py_trees.logging.Level.INFO  # Set the logging level used by py_trees

    # Create the blackboard data structure
    py_trees.blackboard.Blackboard.enable_activity_stream(maximum_size=100)  # Keep first n entries in activity stream
    blackboard = py_trees.blackboard.Client(name="Setup")  # Create instance of the blackboard
    blackboard.register_key(key="agent", access=py_trees.common.Access.WRITE)  # Register agent fields on the blackboard
    blackboard.register_key(key="new_goal_pos", access=py_trees.common.Access.WRITE)  # Register return field for the BT

    # Assign initial values for blackboard parameters
    agent = AgentDummy()  # Create dummy agent object to test the blackboard
    blackboard.agent = agent  # Assign the agent object to the blackboard
    blackboard.new_goal_pos = agent.goal_pos  # Initialise the new_goal_pos parameter

    root = root()  # Create the root behaviour
    behaviour_tree = py_trees.trees.BehaviourTree(root=root)  # Create the behaviour tree object
    behaviour_tree.setup(timeout=0.05)

    console = console()  # Initialise the console object

    def pre_tick(tree):
        """Executes before each tick of the behaviour tree."""
        while not bt_send.empty():
            blackboard.agent = bt_send.get()  # Update the blackboard with the latest agent data

    def post_tick(tree):
        """Executes at the end of each tick."""
        bt_return.put(blackboard.new_goal_pos)

        # Console output
        console.clear()
        console.addstr('============= Behaviour Tree Status =============\n\n')
        console.addstr(py_trees.display.unicode_tree(root=tree.root, show_status=True))
        console.refresh()

    try:
        behaviour_tree.tick_tock(
            period_ms=1,
            number_of_iterations=py_trees.trees.CONTINUOUS_TICK_TOCK,
            pre_tick_handler=pre_tick,
            post_tick_handler=post_tick
        )
    except KeyboardInterrupt:
        behaviour_tree.interrupt()


def main(render=True):
    py_trees.logging.level = py_trees.logging.Level.INFO  # Set the logging level used by py_trees

    bt_send = Queue()  # A queue to receive data from the agent
    bt_return = Queue()  # A queue to send output data from the BT back to the agent

    py_trees.blackboard.Blackboard.enable_activity_stream(maximum_size=100)  # Keep first n entries in activity stream
    blackboard = py_trees.blackboard.Client(name="Setup")  # Create instance of the blackboard
    blackboard.register_key(key="agent", access=py_trees.common.Access.WRITE)
    blackboard.register_key(key="new_goal_pos", access=py_trees.common.Access.WRITE)

    agent = AgentDummy()  # Create dummy agent object to test the blackboard
    blackboard.agent = agent  # Assign the agent object to the blackboard
    blackboard.new_goal_pos = 0  # Initialise the new_goal_pos parameter

    root = highway_drive()

    # Rendering
    if render:
        # py_trees.display.render_dot_tree(root, with_blackboard_variables=True)
        dottree = py_trees.display.dot_tree(root, with_blackboard_variables=True, with_qualified_names=False)
        dottree.write_svg("logic_diagram/AutonomousHighwayDriving.svg")
        sys.exit()

    behaviour_tree = py_trees.trees.BehaviourTree(root=root)
    print(py_trees.display.unicode_tree(root=root))
    behaviour_tree.setup(timeout=0.05)

    def pre_tick(tree):
        print("---------new tick---------\n")
        print("Executing pre-tick handler. Getting latest agent data...")
        while not bt_send.empty():
            blackboard.agent = bt_send.get()  # Update the blackboard with the latest agent data

    def post_tick(tree):
        print("Executing post-tick handler.")
        bt_return.put(blackboard.new_goal_pos)

        print(py_trees.display.unicode_tree(root=tree.root, show_status=True))  # Print the unicode BT to the console
        print("--------------------------\n")
        print(py_trees.display.unicode_blackboard())  # Display blackboard data
        print("--------------------------\n")
        # Display which behaviours have read/write permissions to data in the blackboard
        print(py_trees.display.unicode_blackboard(display_only_key_metadata=True))
        print("--------------------------\n")
        print(py_trees.display.unicode_blackboard_activity_stream())
        print("--------------------------\n")

    try:
        behaviour_tree.tick_tock(
            period_ms=500,
            number_of_iterations=py_trees.trees.CONTINUOUS_TICK_TOCK,
            pre_tick_handler=pre_tick,
            post_tick_handler=post_tick
        )
    except KeyboardInterrupt:
        behaviour_tree.interrupt()


if __name__ == "__main__":
    main(render=True)
