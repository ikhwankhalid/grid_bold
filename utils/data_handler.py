# %%
import os
import numpy as np
from numpy.lib.function_base import diff
import scipy.io
import matplotlib.pyplot as plt


def load_data():
    """
    Loads all available subject path data from the data/paths directory
    and returns the data as a dictionary.

    Returns:
        dict: A dictionary with {"subject": "trajectory [t, x, y]"}
    """
    data = {}

    # get the location of the data
    loc = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            r"data/paths"
        )
    )

    # fill dictionary
    for subdir, dirs, files in os.walk(loc):
        for file in files:
            loc = os.path.join(subdir, file)
            mat = scipy.io.loadmat(loc)["txy"]
            basename = os.path.basename(os.path.dirname(loc))
            data[basename] = mat

    # TH data needs to be centered
    for name, trajec in data.items():
        if name.startswith("TH"):
            data[name][:, 1:] -= np.mean(data[name][:, 1:], axis=0)

    return data


def clean_data(data: dict) -> dict:
    """
    [summary]

    Args:
        data (dict): [description]

    Returns:
        dict: [description]
    """
    for name, traj in data.items():
        traj = data[name]
        if len(np.where(
            np.all(np.diff(traj, axis=0)[:, 1:] == [0., 0.], axis=1)
        )) > 0:
            idx = np.where(
                np.all(np.diff(traj, axis=0)[:, 1:] == [0., 0.], axis=1)
            )[0] + 1
            traj_cleaned = np.delete(traj, idx, axis=0)
            data[name] = traj_cleaned

    return data


def get_dirs(data: dict) -> dict:
    """
    Takes a dictionary of trajectories, calculates direction of movement,
    and returns a new dictionary with direction data.

    Args:
        dict: A dictionary with {"subject": "trajectory [t, x, y]"}

    Returns:
        dict: A dictionary with {"subject": "trajectory [t, x, y, dirs]"}
    """
    for name, pos in data.items():
        dr = np.zeros((len(pos), 2))
        dr[1:, :] = np.diff(pos[:, 1:], axis=0)
        dirs = np.arctan2(dr[:, 1], dr[:, 0]).reshape(-1, 1)
        data[name] = np.hstack((pos, dirs))

    return data


# %%
