import py_trees
# import py_trees.behaviours
# import py_trees.display
import sys
import time


def create_root() -> py_trees.behaviour.Behaviour:
    """Create the root behaviour and its subtree.
    Returns:
        The root behaviour"""
    root = py_trees.composites.Selector(name="Selector", memory=False)
    ffs = py_trees.behaviours.StatusSequence(
        name="FFS",
        sequence=[
            py_trees.common.Status.FAILURE,
            py_trees.common.Status.FAILURE,
            py_trees.common.Status.SUCCESS,
        ],
        eventually=py_trees.common.Status.SUCCESS,
    )
    always_running = py_trees.behaviours.Running(name="Running")
    root.add_children([ffs, always_running])
    return root


def main() -> None:
    """Entry point for the demo script."""
    py_trees.logging.level = py_trees.logging.Level.DEBUG
    root = create_root()

    if False:
        # VisibilityLevel: [ALL, DETAIL, COMPONENT, BIG_PICTURE]
        dottree = py_trees.display.render_dot_tree(root, visibility_level=py_trees.common.VisibilityLevel.DETAIL)
        # dottree.write_png("BT.png")
        sys.exit()

    # Execute
    root.setup_with_descendants()
    for i in range(1, 4):
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
