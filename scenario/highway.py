FPS = 480  # Target frame rate for the simulation
WINDOW_WIDTH = 1536  # Width of the Pygame window (default=1536)
WINDOW_HEIGHT = 120  # Height of the Pygame window (default=120)
BACKGROUND_IMAGE = 'highway.jpg'  # Background image for the Pygame window

DEAD_ZONE = 100  # CollisionHistory occurring before this line are not counted

# y position of each lane
# Driving on the right
LANES = {
    0: 19,
    1: 32,
    2: 44,
    3: 75,
    4: 89,
    5: 102
}

# Driving on the left
# LANES = {
#     5: 19,
#     4: 32,
#     3: 44,
#     2: 75,
#     1: 89,
#     0: 102
# }

# The velocity corresponding to each lane (mph)
LANE_VELOCITIES = {
    0: 50,
    1: 60,
    2: 70,
    3: 70,
    4: 60,
    5: 50,
}

SPEED_LIMIT = 70
