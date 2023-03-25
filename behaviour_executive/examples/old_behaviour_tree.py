import py_trees
import multiprocessing
import multiprocessing.connection
import atexit
import time
import sys

import os

# Conditions


class Condition(py_trees.behaviour.Behaviour):
    """A generic behaviour that checks a condition."""
    def __init__(self, name: str = "Condition"):
        """Configure the name of the behaviour"""
        super(Condition, self).__init__(name)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def setup(self) -> None:
        """No delayed initialisation required"""
        self.logger.debug("%s.setup()" % self.__class__.__name__)

    def initialise(self) -> None:
        self.logger.debug("%s.initialise()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        """Check the condition and decide on a new status"""
        self.condition = False

        if self.condition:
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example"""
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


class ClearRoad(py_trees.behaviour.Behaviour):
    """Condition: Checks if the road ahead is clear."""
    def __init__(self, name: str = "Clear road"):
        """Configure the name of the behaviour"""
        super(ClearRoad, self).__init__(name)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def setup(self) -> None:
        """No delayed initialisation required"""
        self.logger.debug("%s.setup()" % self.__class__.__name__)

    def initialise(self) -> None:
        self.logger.debug("%s.initialise()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        """Check the condition and decide on a new status"""
        self.condition = True

        if self.condition:
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example"""
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


class VehicleApproaching(py_trees.behaviour.Behaviour):
    """Condition: Checks for vehicles approaching from behind."""
    def __init__(self, name: str = "Vehicle approaching"):
        """Configure the name of the behaviour"""
        super(VehicleApproaching, self).__init__(name)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def setup(self) -> None:
        """No delayed initialisation required"""
        self.logger.debug("%s.setup()" % self.__class__.__name__)

    def initialise(self) -> None:
        self.logger.debug("%s.initialise()" % self.__class__.__name__)

    def update(self) -> py_trees.common.Status:
        """Check the condition and decide on a new status"""
        self.condition = True

        if self.condition:
            new_status = py_trees.common.Status.SUCCESS
        else:
            new_status = py_trees.common.Status.RUNNING
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example"""
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )

# Actions


def planning(pipe_connection: multiprocessing.connection.Connection) -> None:
    """Emulate a (potentially) long-running process.
    Args:
        pipe_connection: connection to the planning process
    """
    idle = True
    percentage_complete = 0
    try:
        while True:
            if pipe_connection.poll():
                pipe_connection.recv()
                percentage_complete = 0
                idle = False
            if not idle:
                percentage_complete += 10
                pipe_connection.send([percentage_complete])
                if percentage_complete == 100:
                    idle = True
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


class Action(py_trees.behaviour.Behaviour):
    def __init__(self, name: str):
        """Configure the name of the behaviour."""
        super(Action, self).__init__(name)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def setup(self, **kwargs: int) -> None:
        """Kickstart the separate process this behaviour will work with.
        Process is usually already running - setup method is only used to verify it exists."""

        self.logger.debug("%s.setup()->connections to an external process" % self.__class__.__name__)

        self.parent_connection, self.child_connection = multiprocessing.Pipe()
        self.planning = multiprocessing.Process(target=planning, args=(self.child_connection,))
        atexit.register(self.planning.terminate)
        self.planning.start()

    def initialise(self) -> None:
        """Reset a counter variable."""
        self.logger.debug("%s.initialise()->sending new goal" % self.__class__.__name__)
        self.parent_connection.send(["new goal"])
        self.percentage_completion = 0

    def update(self) -> py_trees.common.Status:
        """Increment the counter, monitor and decide new status."""
        new_status = py_trees.common.Status.RUNNING
        if self.parent_connection.poll():
            self.percentage_completion = self.parent_connection.recv().pop()
            if self.percentage_completion == 100:
                new_status = py_trees.common.Status.SUCCESS
        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = "Processing finished"
            self.logger.debug("%s.update()[%s->%s][%s]" % (self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            self.feedback_message = "{0}%".format(self.percentage_completion)
            self.logger.debug("%s.update()[%s][%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example."""
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


# Composites
def create_selector(children: [py_trees.behaviour.Behaviour]) -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
        Returns:
            The root behaviour"""
    selector = py_trees.composites.Selector(name="Selector", memory=False)
    selector.add_children([*children])
    return selector


def create_sequence(children: [py_trees.behaviour.Behaviour]) -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
        Returns:
            The root behaviour"""
    sequence = py_trees.composites.Sequence(name="Sequence", memory=False)
    sequence.add_children([*children])
    return sequence


def main(render=True):
    py_trees.logging.level = py_trees.logging.Level.INFO

    # Left sub-tree -----------------
    # Conditions
    clear_road = Condition(name="Clear Road")
    vehicle_approaching = Condition(name="Vehicle Approaching")
    right_lane_free = Condition(name="Right Lane Free")

    # Decorators
    inverter_vehicle_approaching = py_trees.decorators.Inverter(
        name="Inverter",
        child=vehicle_approaching
        )
    inverter_right_lane_free = py_trees.decorators.Inverter(
        name="Inverter",
        child=right_lane_free
    )

    # Actions
    lane_change_right = Action(name="Lane Change Right")
    free_ride = Action(name="Free Ride")

    # Composites
    yield_to_vehicle = create_selector([inverter_vehicle_approaching, inverter_right_lane_free, lane_change_right])
    free_ride_seq = create_sequence([clear_road, yield_to_vehicle, free_ride])

    # Right sub-tree ---------
    # Conditions
    slow_vehicle = Condition(name="Slow Vehicle")
    left_lane_free = Condition(name="Left Lane Free")

    # Actions
    lane_change_left = Action(name="Lane Change Left")
    follow = Action(name="Follow")

    # Composites
    overtake = create_sequence([slow_vehicle, left_lane_free, lane_change_left])
    follow_sel = create_selector([overtake, follow])

    # Create root
    root = create_selector([free_ride_seq, follow_sel])

    # Rendering
    if render:
        dottree = py_trees.display.dot_tree(root)
        dottree.write_svg("logic_diagram/AutonomousHighwayDriving.svg")
        sys.exit()

    def print_tree(tree):
        print(py_trees.display.unicode_tree(root=tree.root, show_status=True))

    behaviour_tree = py_trees.trees.BehaviourTree(root=root)
    print(py_trees.display.unicode_tree(root=root))
    behaviour_tree.setup(timeout=15)

    try:
        behaviour_tree.tick_tock(
            period_ms=500,
            number_of_iterations=py_trees.trees.CONTINUOUS_TICK_TOCK,
            pre_tick_handler=None,
            post_tick_handler=print_tree
        )
    except KeyboardInterrupt:
        behaviour_tree.interrupt()

    # clear = lambda: os.system('cls')
    #
    # # Execute
    # root.setup_with_descendants()  # Call setup() method of all children
    # for i in range(1, 40):
    #     try:
    #         # print("\033[H\033[J", end="")
    #         clear()
    #         print("\n--------- Tick {0} ---------\n".format(i))
    #         root.tick_once()
    #         print("\n")
    #         print(py_trees.display.unicode_tree(root=root, show_status=True))
    #         time.sleep(1.0)
    #     except KeyboardInterrupt:
    #         break
    # print("\n")
    #


if __name__ == "__main__":
    main(render=False)
