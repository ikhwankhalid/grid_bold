"""
This module contains functions to load and clean the human experimental data
from the data/paths directory.
"""
import os
import numpy as np
import scipy.io
import random


def load_data():
    """
    Loads all available subject path data from the data/paths directory
    and returns the data as two dictionaries, one for files starting with "TH"
    and one for files starting with "OF".

    Returns:
        two dictionaries with {"subject": "trajectory [t, x, y]"}
    """
    th_data = {}
    of_data = {}

    # get the location of the data
    loc = os.path.realpath(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            r"data/paths"
        )
    )

    # fill dictionaries
    for subdir, dirs, files in os.walk(loc):
        for file in files:
            if file.endswith(".mat"):
                loc = os.path.join(subdir, file)
                mat = scipy.io.loadmat(loc)["txy"]
                basename = os.path.basename(os.path.dirname(loc))

                if basename.startswith("TH"):
                    # TH data needs to be centered
                    mat[:, 1:] -= np.mean(mat[:, 1:], axis=0)
                    th_data[basename] = mat
                elif basename.startswith("OF"):
                    mat[:, 1:] /= 60.
                    of_data[basename] = mat

    return th_data, of_data


def clean_data(data: dict) -> dict:
    """
    Clean the input data by removing sub-sequential points with zero positional
    changes.

    Assumes the input data is a dictionary where each key corresponds to a
    unique identifier (e.g., name), and its value is a numpy array representing
    a trajectory. Each row in the numpy array represents a point in the
    trajectory, with the first column typically being the time and subsequent
    columns representing the spatial coordinates.

    This function iterates through the dictionary, and for each trajectory, it
    detects subsequences of points where the spatial differences (ignoring the
    first column, usually time) between consecutive points are [0.0, 0.0].
    Such subsequences (excluding the first point of subsequence) are deemed
    redundant and are removed from the trajectory, making the trajectory
    representation more compact and eliminating stationary points, assuming no
    movement as a lack of change in spatial coordinates.

    Parameters:
    - data (dict): A dictionary where keys are identifiers and values are numpy
        arrays representing trajectories.

    Returns:
    - dict: A dictionary with the same structure as the input, where each
        trajectory has been cleaned by removing redundant subsequences of
        stationary points.

    Note:
    - The function mutates the input data, so the original data structure is
        modified.
    - The first column of the numpy arrays is not considered when detecting
        stationary points, making it suitable for trajectories where the first
        column is a temporal component.
    """
    for name, traj in data.items():
        traj = data[name]
        if len(
            np.where(
                np.all(np.diff(traj, axis=0)[:, 1:] == [0., 0.], axis=1)
            )
        ) > 0:
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
        dirs[0] = 2 * np.pi * random.random() - np.pi
        data[name] = np.hstack((pos, dirs))

    return data
