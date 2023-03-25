# Imports
import py_trees
import py_trees.console as console
import sys
import operator

import os
# os.environ["PYTHONIOENCODING"] = 'utf-8'
# os.environ["PYTHONLEGACYWINDOWSSTDIO"] = 'utf-8'

# sys.stdin.reconfigure(encoding="utf-8")
# sys.stdout.reconfigure(encoding="utf-8")


class Nested(object):
    """A more complex object to interact with the blackboard."""

    def __init__(self) -> None:
        """Initialise variables to some arbitrary defaults."""
        self.foo = "bar"

    def __str__(self) -> str:
        return str({"foo": self.foo})


class BlackboardWriter(py_trees.behaviour.Behaviour):
    """Write some more interesting / complex types to the blackboard."""

    def __init__(self, name: str):
        """Set up the blackboard.
        Args:
            name: behaviour name
        """
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key="dude", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="spaghetti", access=py_trees.common.Access.WRITE
        )

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self) -> py_trees.common.Status:
        """Write a dictionary to the blackboard.
        This behaviour always returns SUCCESS.
        """
        self.logger.debug("%s.update()" % (self.__class__.__name__))
        try:
            _ = self.blackboard.dude
        except KeyError:
            pass
        try:
            _ = self.blackboard.dudette
        except AttributeError:
            pass
        try:
            self.blackboard.dudette = "Jane"
        except AttributeError:
            pass
        self.blackboard.spaghetti = {"type": "Carbonara", "quantity": 1}
        self.blackboard.spaghetti = {"type": "Gnocchi", "quantity": 2}
        try:
            self.blackboard.set(
                "spaghetti", {"type": "Bolognese", "quantity": 3}, overwrite=False
            )
        except AttributeError:
            pass
        return py_trees.common.Status.SUCCESS


class ParamsAndState(py_trees.behaviour.Behaviour):
    """Parameter and state storage on the blackboard.
    This behaviour demonstrates the usage of namespaces and
    multiple clients to perform getting and setting of
    parameters and state in a concise and convenient manner.
    """

    def __init__(self, name: str):
        """Set up separate blackboard clients for parameters and state.
        Args:
            name: behaviour name"""
        super().__init__(name=name)
        # Namespaces can include the separator or leave it out
        # Can alse be nested (E.g. /agent/state, /agent/parameters)
        self.parameters = self.attach_blackboard_client("Params", "parameters")
        self.state = self.attach_blackboard_client("State", "state")
        self.parameters.register_key(
            key="default_speed", access=py_trees.common.Access.READ
        )
        self.state.register_key(
            key="current_speed", access=py_trees.common.Access.WRITE
        )

    def initialise(self) -> None:
        """Initialise speed from the stored parameter variable on the blackboard."""
        try:
            self.state.current_speed = self.parameters.default_speed
        except KeyError as e:
            raise RuntimeError(
                "parameter 'default_speed' not found [{}]".format(str(e))
            )

    def update(self) -> py_trees.common.Status:
        """
        Check speed and either increment, or complete if it has reached a threshold.
        Returns:
            RUNNING if incrementing, SUCCESS otherwise.
        """
        if self.state.current_speed > 40.0:
            return py_trees.common.Status.SUCCESS
        else:
            self.state.current_speed += 1.0
            return py_trees.common.Status.RUNNING


def create_root() -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
    Returns:
        the root behaviour"""
    root = py_trees.composites.Sequence(name="Blackboard Demo", memory=False)
    set_blackboard_variable = py_trees.behaviours.SetBlackboardVariable(
        name="Set Nested",
        variable_name="nested",
        variable_value=Nested(),
        overwrite=True,
    )
    write_blackboard_variable = BlackboardWriter(name="Writer")
    check_blackboard_variable = py_trees.behaviours.CheckBlackboardVariableValue(
        name="Check Nested Foo",
        check=py_trees.common.ComparisonExpression(
            variable="nested.foo", value="bar", operator=operator.eq
        ),
    )
    params_and_state = ParamsAndState(name="ParamsAndState")
    root.add_children(
        [
            set_blackboard_variable,
            write_blackboard_variable,
            check_blackboard_variable,
            params_and_state,
        ]
    )
    return root


def main(render) -> None:
    """Entry point for the demo script."""
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    py_trees.blackboard.Blackboard.enable_activity_stream(maximum_size=100)
    blackboard = py_trees.blackboard.Client(name="Configuration")
    blackboard.register_key(key="dude", access=py_trees.common.Access.WRITE)
    blackboard.register_key(
        key="/parameters/default_speed", access=py_trees.common.Access.EXCLUSIVE_WRITE
    )
    blackboard.dude = "Bob"
    blackboard.parameters.default_speed = 30.0

    root = create_root()

    # Rendering
    if render:
        py_trees.display.render_dot_tree(root, with_blackboard_variables=True)
        sys.exit()

    # Execute
    root.setup_with_descendants()
    unset_blackboard = py_trees.blackboard.Client(name="Unsetter")
    unset_blackboard.register_key(key="foo", access=py_trees.common.Access.WRITE)
    print("\n--------- Tick 0 ---------\n")
    root.tick_once()
    print("\n")
    print(py_trees.display.unicode_tree(root, show_status=True))
    print("--------------------------\n")
    print(py_trees.display.unicode_blackboard())
    print("--------------------------\n")
    print(py_trees.display.unicode_blackboard(display_only_key_metadata=True))
    print("--------------------------\n")
    unset_blackboard.unset("foo")
    print(py_trees.display.unicode_blackboard_activity_stream())


if __name__ == "__main__":
    main(True)
