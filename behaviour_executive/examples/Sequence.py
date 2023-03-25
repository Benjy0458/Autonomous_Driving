import py_trees
import time
import sys


def create_root() -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
    Returns:
        The root behaviour"""
    root = py_trees.composites.Sequence(name="Sequence", memory=False)
    for action in ["Action 1", "Action 2", "Action 3"]:
        rssss = py_trees.behaviours.StatusSequence(
            name=action,
            sequence=[
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.SUCCESS,
            ],
            eventually=py_trees.common.Status.SUCCESS,
        )
        root.add_child(rssss)
    return root


def main() -> None:
    """Entry point for the demo script."""
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    root = create_root()

    # Rendering
    if True:
        dottree = py_trees.display.dot_tree(root)
        dottree.write_svg("logic_diagram/root.svg")
        sys.exit()

    # Execute
    root.setup_with_descendants()
    for i in range(1, 6):
        try:
            print("\n--------- Tick {0} ---------\n".format(i))
            root.tick_once()
            print("\n")
            print(py_trees.display.unicode_tree(root=root, show_status=True))
            time.sleep(1.0)
        except KeyboardInterrupt:
            break
    print("\n")


if __name__ == "__main__":
    main()