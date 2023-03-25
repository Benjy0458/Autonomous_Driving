import py_trees
import time


class Counter(py_trees.behaviour.Behaviour):
    """Simple counting behaviour.
    - Increment a counter from zero on each tick
    - Finish with success if counter reaches 3
    - Reset counter in initialise() method
    """
    def __init__(self, name: str = "Counter"):
        """Configure the name of the behaviour"""
        super(Counter, self).__init__(name)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def setup(self) -> None:
        """No delayed initialisation is required for this example."""
        self.logger.debug("%s.setup()" % self.__class__.__name__)

    def initialise(self) -> None:
        """Reset a counter variable."""
        self.logger.debug("%s.initialise()" % self.__class__.__name__)
        self.counter = 0

    def update(self) -> py_trees.common.Status:
        """Increment the counter and decide on a new status."""
        self.counter += 1
        new_status = (
            py_trees.common.Status.SUCCESS
            if self.counter == 3
            else py_trees.common.Status.RUNNING
        )
        if new_status == py_trees.common.Status.SUCCESS:
            self.feedback_message = (
                "counting...{0} - ".format(self.counter)
            )
        else:
            self.feedback_message = "still counting"
        self.logger.debug(
            "%s.update()[%s->%s][%s]"
            % (self.__class__.__name__, self.status, new_status, self.feedback_message)
        )
        return new_status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up in this example."""
        self.logger.debug(
            "%s.terminate()[%s->%s]"
            % (self.__class__.__name__, self.status, new_status)
        )


def main() -> None:
    """Entry point for the demo script."""

    py_trees.logging.level = py_trees.logging.Level.DEBUG

    counter = Counter()
    counter.setup()
    print("")
    try:
        for _ in range(0, 7):
            counter.tick_once()
            print("")
            time.sleep(0.5)
        print("\n")
    except KeyboardInterrupt:
        print("")
        pass


if __name__ == "__main__":
    main()
