import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import utils

from pprint import pprint


def plot_human_traj(human_traj_data, observed_tracklet_length):
    plt.scatter(human_traj_data[observed_tracklet_length:, 1], human_traj_data[observed_tracklet_length:, 2], marker='o', alpha=1, color="r", s=50, label="Ground truth")


def plot_all_predicted_trajs(total_predicted_motion_list, observed_tracklet_length):
    for predicted_traj in total_predicted_motion_list:
        shape = predicted_traj.shape
        (u, v) = utils.pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=50)
        # plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=50)
        for i in range(observed_tracklet_length, shape[0]):
            total = shape[0] - observed_tracklet_length
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="b", marker="o", alpha=1-(i-observed_tracklet_length)/total, s=50)


def plot_cliff_map(cliff_map_data):
    (u, v) = utils.pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    color = cliff_map_data[:, 2]
    plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color, alpha=0.7, cmap="hsv")

