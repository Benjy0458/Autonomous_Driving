SUCCESS = "Success"
RUNNING = "Running"
FAILURE = "Failure"


class BehaviourTree:
    pass


class Nodes(BehaviourTree):
    def __init__(self):
        self.status = None

    def tick(self):
        pass


class Control(Nodes):
    def __init__(self, *children):
        super().__init__()
        self.children = children

    def __str__(self):
        return str([*map(str, self.children)])  # Equiv. to: str([str(child) for child in self.children])


class Execution(Nodes):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name


class Sequence(Control):
    def tick(self):
        for child in self.children:
            child_status = child.tick()
            if child_status != SUCCESS:
                self.status = child_status
                return self.status

        self.status = SUCCESS
        return self.status


class Fallback(Control):
    def tick(self):
        for child in self.children:
            child_status = child.tick()
            if child_status != FAILURE:
                self.status = child_status
                return self.status

        self.status = FAILURE
        return self.status


class Parallel(Control):
    def __init__(self, *children):
        super().__init__(*children)

    def tick(self):
        success_count = 0
        failure_count = 0
        for child in self.children:
            child_status = child.tick()
            if child_status == SUCCESS:
                success_count += 1
            elif child_status == FAILURE:
                failure_count += 1

        # Change this-> Should succeed if at least m successes
        if success_count == len(self.children):
            self.status = SUCCESS
        elif failure_count == len(self.children):
            self.status = FAILURE
        else:
            self.status = RUNNING

        return self.status


class Decorator(Nodes):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def tick(self):
        pass


class Inverter(Decorator):
    def tick(self):
        child_status = self.child.tick()
        if child_status == SUCCESS:
            self.status = FAILURE
        elif child_status == FAILURE:
            self.status = SUCCESS
        else:
            self.status = child_status

        return self.status


class Action(Execution):
    def __init__(self, name, action_func):
        super().__init__(name)
        self.action_func = action_func

    def tick(self):
        self.status = self.action_func()
        return self.status


class Condition(Execution):
    def __init__(self, name, condition_func):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self):
        self.status = SUCCESS if self.condition_func() else FAILURE

        return self.status


def example():
    place_ball = Action("Place ball", "Failure")
    find_ball = Action("Find ball", "Success")

    approach_ball = Action("Approach ball", "Success")
    grasp_ball = Action("Grasp ball", "Failure")

    ball_close = Condition("Ball close?", True)
    ball_grasped = Condition("Ball grasped?", False)

    approach = Fallback(ball_close, approach_ball)
    grasp = Fallback(ball_grasped, grasp_ball)
    pick_ball = Sequence(approach, grasp)

    BT = Sequence(find_ball, pick_ball, place_ball)
    print(BT)

    # print(ball_close.tick())
    # print(approach_ball.tick())

    BT.tick()
    print(BT.tick())


def example2():
    def move_forward():
        return SUCCESS

    def shoot():
        return SUCCESS

    def is_enemy_near():
        return True

    def is_cover_nearby():
        return False

    # Create nodes
    action_node1 = Action("Move forward", move_forward)
    action_node2 = Action("Shoot", shoot)
    condition_node1 = Condition("Is enemy near", is_enemy_near)
    condition_node2 = Condition("Is cover nearby", is_cover_nearby)
    inverter_node = Inverter(condition_node2)
    parallel_node = Parallel(action_node1, inverter_node, condition_node1, action_node2)
    parallel_node.tick()
    print(parallel_node.status)


def bt():
    # Define leaf nodes
    clear_road = Condition("Clear road", SUCCESS)
    right_lane_free = Condition("Right lane free", SUCCESS)
    slow_vehicle = Condition("Slow vehicle", SUCCESS)
    left_lane_free = Condition("Left lane free", SUCCESS)

    lane_change_right = Action("Lane change right", SUCCESS)
    free_ride = Action("Free ride", SUCCESS)
    lane_change_left = Action("Lane change left", SUCCESS)
    follow = Action("Follow", SUCCESS)

    # Define control nodes
    dec = Decorator(right_lane_free, "invert")
    fall = Fallback(dec, lane_change_right)
    branch1 = Sequence(clear_road, fall, free_ride)

    seq = Sequence(slow_vehicle, left_lane_free, lane_change_left)
    branch2 = Fallback(seq, follow)
    root = Fallback(branch1, branch2)

    return root


def main():
    example2()
    # print(1 >> 0)
    # print(bt())
    # bt().tick()

    # act = Action("Action", SUCCESS)
    # dec = Decorator(act, "invert")
    # root = Sequence(dec)
    # print(root.tick())
    # print(root)


if __name__ == "__main__":
    main()
