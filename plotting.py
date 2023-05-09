'''
This module provides some helper functions for real time plotting of the simulation data.
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
file_handler = logging.FileHandler(f'LogFiles/{__name__}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)  # Add file handler to the logger


def animate(i, ax, queue, data=[]):
    ((ax1, ax3), (ax2, ax4)) = ax

    while not queue.empty():
        d = queue.get()
        if d:
            data.append(queue.get())
        else:
            return

    if len(data) > 500:
        data = data[-499:]

    (t, xte, phi, delta, vel, acc) = list(map(list, zip(*data)))

    ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear()  # Clear axes
    ax1.set_ylim([-50, 50]), ax2.set_ylim([-50, 50]), ax3.set_ylim([20, 90]), ax4.set_ylim([-9.5, 5.5])  # Set axis limits

    ax1.plot(t, xte, c='b', label='xte', linewidth=0.2)  # Plot data
    ax1.plot(t, phi, c='g', label='phi', linewidth=0.2)  # Plot data

    ax2.plot(t, delta, c='r', label='delta', linewidth=0.2)
    ax1.set_title('Lateral controller')
    ax1.set_ylabel('State', fontsize=10)  # Set axes labels
    ax2.set_ylabel('Steering angle', color='black', fontsize=10)
    ax2.set_xlabel('Elapsed time (s)', fontsize=10)
    ax1.legend(loc='upper right')

    ax3.plot(t, vel, c='b', label='vel', linewidth=0.2)  # Plot data
    ax4.plot(t, acc, c='r', label='acc', linewidth=0.2)  # Plot data
    ax3.set_title('IDM controller')
    ax3.set_ylabel('Velocity (mph)', fontsize=10)
    ax4.set_xlabel('Elapsed time (s)', fontsize=10)
    ax4.set_ylabel('Acceleration (m/s^2)', fontsize=10)

    plt.tight_layout()
    plt.rcParams['font.size'] = 10  # Set tick size


def live_graph(queue, close):
    style.use('fivethirtyeight')

    fig, ax = plt.subplots(nrows=2, ncols=2)
    thismanager = plt.get_current_fig_manager()
    thismanager.set_window_title("Agent Profile")
    x, y = 1530 - 410, 188
    thismanager.window.wm_geometry("+%d+%d" % (x, y))

    ani = animation.FuncAnimation(fig, animate, interval=10, fargs=(ax, queue), cache_frame_data=False)  # Update the figure window every 10 ms.
    plt.show(block=False)  # Display the plots

    while close.empty():
        plt.pause(1)

    logger.debug("Closing the live plot window")
    try:
        ani.event_source.stop()
        del ani
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Encountered {e} attempting to close the live plot.")
    else:
        logger.debug("Succesfully closed the window")
