import py_trees
import multiprocessing
import multiprocessing.connection
import time
import atexit

import py_trees.logging


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
            time.sleep(0.5)
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


def main() -> None:
    """Entry point for the demo script."""

    py_trees.logging.level = py_trees.logging.Level.DEBUG

    action = Action(name="Action")
    action.setup()
    try:
        for _ in range(0, 14):
            action.tick_once()
            time.sleep(0.5)
        print("\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
