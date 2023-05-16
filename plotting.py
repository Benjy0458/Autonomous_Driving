"""This module provides some helper functions for real time plotting of the simulation data."""

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


def animate(i, ax, queue, t=[], xte=[], phi=[], delta=[], vel=[], acc=[], data=[]):
    (ax1, ax2, ax3) = ax

    while not queue.empty():
        data = queue.get()
        for i, var in enumerate([t, xte, phi, delta, vel, acc]):
            var.append(data[i])

    while len(t) > 1500:
        del t[0], xte[0], phi[0], delta[0], vel[0], acc[0]

    ax1.clear(), ax2.clear(), ax3.clear()  # Clear axes
    ax1.set_ylim([-5, 5]), ax2.set_ylim([20, 90]), ax3.set_ylim([-9.5, 5.5])

    ax1.plot(t, xte, c='b', label='xte (m)', linewidth=0.2)  # Plot data
    ax1.plot(t, phi, c='g', label='phi (rad)', linewidth=0.2)  # Plot data

    ax1.plot(t, delta, c='r', label='delta (rad)', linewidth=0.2)

    ax2.plot(t, vel, c='b', label='vel', linewidth=0.2)  # Plot data
    ax3.plot(t, acc, c='r', label='acc', linewidth=0.2)  # Plot data

    ax1.set_title('Lateral controller')
    ax1.set_ylabel('State', fontsize=10)  # Set axes labels
    ax2.set_ylabel('Steering angle', color='black', fontsize=10)
    ax2.set_xlabel('Elapsed time (s)', fontsize=10)
    ax1.legend(loc='upper right')

    ax2.set_title('IDM controller')
    ax2.set_ylabel('Velocity (mph)', color='blue', fontsize=10)
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('Acceleration (m/s^2)', color='red', fontsize=10)
    ax3.grid(False)

    plt.tight_layout()
    plt.rcParams['font.size'] = 10  # Set tick size


def live_graph(queue, close):
    style.use('fivethirtyeight')

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    (ax1, ax2) = ax
    ax3 = ax2.twinx()
    ax = (ax1, ax2, ax3)

    thismanager = plt.get_current_fig_manager()
    thismanager.set_window_title("Agent Profile")
    x, y = 1530 - 410, 188
    thismanager.window.wm_geometry("+%d+%d" % (x, y))

    ani = animation.FuncAnimation(fig, animate, interval=0, fargs=(ax, queue), cache_frame_data=False)  # Update the figure window every 10 ms.
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
