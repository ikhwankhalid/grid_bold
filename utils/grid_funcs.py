import numpy as np
import matplotlib.pyplot as plt
import os
import settings
import scipy.special as sc
from numba import njit, jit
from scipy import interpolate
from tqdm import tqdm
from utils.data_handler import load_data, get_dirs, clean_data
from typing import Tuple
from utils.utils import (
    convert_to_rhombus,
    grid, adap_euler,
    grid_meanfr
)


@jit
def circle_bound(
    x: float,
    y: float,
    bound: float,
    loc: float = np.array([0, 0])
) -> Tuple[float, float, float]:
    """
    if r = sqrt(x**2 + y**2) is larger than bound, returns the distance in x,
    y, and total distance r that the point (x, y) is outside the boundary.
    Otherwise, returns a tuple of three zeroes (0, 0, 0) meaning that the point
    (x, y) lies within the circular boundary.

    Parameters
    ----------
        x:      agent's position in x
        y:      agent's position in y
        bound:  radius to the circular boundary
        loc:    center of the circular boundary. Defaults to zero.

    Returns
    ----------
        x_over: the distance in x that the agent has overstepped the
                boundary
        y_over: the distance in y that the agent has overstepped the
                boundary
    """
    # Initialize overstep values
    x_over = 0
    y_over = 0
    r_over = 0

    # Check if point is outside a circle of radius 'bound'
    if np.sqrt((x - loc[0])**2 + (y - loc[1])**2) > bound:
        # If yes, find how much the point (x,y) lies outside the boundary
        r_over = np.sqrt(
            (x - loc[0])**2 + (y - loc[1])**2
        ) - bound
        phi = np.arctan2(y, x)
        x_over = r_over * np.cos(phi)
        y_over = r_over * np.sin(phi)

    return x_over, y_over, r_over


@jit
def square_bound(
    x: float,
    y: float,
    bound: float,
    loc: float = np.array([0, 0])
) -> Tuple[float, float, float]:
    """
    if (x, y) is outside of a square with side length bound * sqrt(pi) / 2
    centered at loc, returns a tuple (x, y, dummy) where x and y are the
    distances that the point (x, y) lies outside the boundary. Otherwise,
    returns a tuple of three zeroes (0, 0, 0) meaning that the point (x, y)
    lies within square boundary.

    Parameters
    ----------
        x:      agent's position in x
        y:      agent's position in y
        bound:  size of environment
        loc:    center of square boundary

    Returns
    ----------
        x_over: the distance in x that the agent has overstepped the
                boundary
        y_over: the distance in y that the agent has overstepped the
                boundary
        dummy:  dummy object just so that the function returns the same length
                of tuple as circle_bound
    """
    # Define 'bound' such that the area would be equal to the circle_bound when
    # using the same value of 'bound'
    bound = bound * np.sqrt(np.pi) / 2

    # Initialize overstep variables
    x_over = 0
    y_over = 0
    dummy = 0

    # Check if the point lies outside the square
    if x < -bound + loc[0]:
        x_over = x + bound + loc[0]
    elif x > bound + loc[0]:
        x_over = x - bound - loc[0]
    if y < -bound + loc[1]:
        y_over = y + bound + loc[1]
    elif y > bound + loc[1]:
        y_over = y - bound - loc[1]

    return x_over, y_over, dummy


def gen_offsets(
    N: int = settings.N,
    kappacl: float = 0.,
    meanoff: tuple = (0., 0.),
) -> Tuple[list, list]:
    """
    Samples random 2-dimensional offsets for N cells from a von Mises
    distribution separately for each dimension.

    Args:
    ----------
        N:          number of cells
        kappacl:    1 / width of the von mises distribution
        meanoff:    mean of the von mises distribution

    Returns:
    ----------
        ox:         list of offsets in x
        oy:         list of offsets in y
    """
    ox, oy = np.random.vonmises(
        2 * np.pi * (meanoff[0] - 0.5), kappacl, int(N)
    ) / 2. / np.pi + 0.5, np.random.vonmises(
        2 * np.pi * (meanoff[1] - 0.5), kappacl, int(N)
    ) / 2. / np.pi + 0.5

    return ox, oy


@jit
def traj_star(
    phbins: int,
    rmax: float,
    dt: float,
    sp: float = settings.speed,
    offset: list = (0., 0.),
    grsc: float = settings.grsc
) -> Tuple[float, float, float, float]:
    """
    Samples a star-like walk trajectory

    Parameters
    ----------
        phbins (int):           number of angles to sample from [0, 2*pi)
        rmax (float):           length of a linear segment in centimeters
        dt (float):             time step size in seconds
        sp (float):             speed of agent in centimeters per second
        offset (list):          starting offset in the unit square of the star
                                in (x, y)

    Returns
    ----------
        t (np.ndarray):         vector of times
        x (np.ndarray):         vector of x coordinates
        y (np.ndarray):         vector of y coordinates
        direc (np.ndarray):     vector of movement directions
    """
    # initializations
    nlin = int(rmax / sp / dt)
    tmax = rmax/sp * phbins
    t = np.linspace(0, tmax, int(tmax / dt))
    angles = np.linspace(- np.pi, np.pi, phbins, endpoint=False)
    direc = np.random.choice(angles, len(angles), replace=False)
    direc = np.repeat(direc, nlin)

    # sanity check
    assert len(direc) == len(t)

    # transform square offsets to unit rhombus
    oxr, oyr = convert_to_rhombus(offset[0], offset[1])

    # initialise trajectory
    x, y = np.zeros((2, len(t)))

    # iterate through movements directions
    for i in range(phbins):
        # starting offset should shift by offset in unit rhombus multiplied by
        # the grid scale
        x[i * nlin] = 0 + oxr * grsc
        y[i * nlin] = 0 + oyr * grsc
        # for each angle, iterate until linear segment has nlin steps
        for j in range(1, nlin):
            x[i * nlin + j] = x[i * nlin + j - 1] + sp\
                * np.cos(direc[i * nlin + j]) * dt
            y[i * nlin + j] = y[i * nlin + j - 1] + sp\
                * np.sin(direc[i * nlin + j]) * dt

    return t, x, y, direc


@jit
def traj_star_delay(
    phbins: int,
    rmax: float,
    dt: float,
    sp: float = settings.speed,
    offset: list = [0., 0.],
    grsc: float = settings.grsc,
    delay: float = 0.
) -> Tuple[float, float, float, float]:
    """
    Samples a star-like walk trajectory

    Parameters
    ----------
        phbins (int):           number of angles to sample from [0, 2*pi)
        rmax (float):           length of a linear segment in centimeters
        dt (float):             time step size in seconds
        sp (float):             speed of agent in centimeters per second
        offset (list):          starting offset in the unit square of the star
                                in (x, y)

    Returns
    ----------
        t (np.ndarray):         vector of times
        x (np.ndarray):         vector of x coordinates
        y (np.ndarray):         vector of y coordinates
        direc (np.ndarray):     vector of movement directions
    """
    # initializations
    delay_steps = int(delay / dt)
    nlin = int(rmax / sp / dt)
    tmax = rmax/sp * phbins
    t = np.linspace(
        0,
        tmax + delay * phbins,
        int(tmax / dt) + delay_steps * phbins
    )   # total duration increased by delay
    angles = np.linspace(- np.pi, np.pi, phbins, endpoint=False)
    direc = np.random.choice(angles, len(angles), replace=False)
    direc = np.repeat(
        direc, nlin + delay_steps
    )  # repeat 'direc' for each step including delay

    # sanity check
    assert len(direc) == len(t)

    # transform square offsets to unit rhombus
    oxr, oyr = convert_to_rhombus(offset[0], offset[1])

    # initialise trajectory
    x, y = np.zeros((2, len(t)))

    # iterate through movements directions
    for i in range(phbins):
        # starting offset should shift by offset in unit rhombus multiplied by
        # the grid scale
        x[i * (nlin + delay_steps)] = 0 + oxr * grsc
        y[i * (nlin + delay_steps)] = 0 + oyr * grsc
        # for each angle, iterate until linear segment has nlin steps
        for j in range(1, nlin + delay_steps):
            if j < delay_steps:  # skip movement calculation during delay
                continue
            x[i * (nlin + delay_steps) + j] = \
                x[i * (nlin + delay_steps) + j - 1] \
                + sp * np.cos(direc[i * (nlin + delay_steps) + j]) * dt
            y[i * (nlin + delay_steps) + j] = \
                y[i * (nlin + delay_steps) + j - 1] \
                + sp * np.sin(direc[i * (nlin + delay_steps) + j]) * dt

    return t, x, y, direc


@jit
def traj_pwl(
    phbins: int,
    rmax: float,
    dt: float,
    sp: float = settings.speed
) -> Tuple[float, float, float, float]:
    """
    Samples a piece-wise linear trajectory by unwrapping a star

    Parameters
    ----------
        phbins (int):           number of angles to sample from [0, 2*pi)
        rmax (float):           length of a linear segment in centimeters
        dt (float):             time step size in seconds
        sp (float):             speed of agent in centimeters per second

    Returns
    ----------
        t (np.ndarray):         vector of times
        x (np.ndarray):         vector of x coordinates
        y (np.ndarray):         vector of y coordinates
        direc (np.ndarray):     vector of movement directions
    """
    nlin = int(rmax / sp / dt)
    tmax = rmax/sp * phbins
    t = np.linspace(0, tmax, int(tmax / dt))
    angles = np.linspace(
        - np.pi, np.pi, phbins, endpoint=False
    )
    direc = np.copy(angles)
    # rng = np.random.default_rng()
    # rng.permutation(direc)
    # print(direc)
    direc = np.random.choice(angles, phbins, replace=False)
    direc = np.repeat(direc, nlin)
    # direc = np.hstack((np.array([0.]), direc))
    assert len(direc) == len(t)

    # initialise trajectory
    x, y = np.zeros((2, len(t)))

    for i in range(0, len(t)):
        if i == 0:
            x0, y0 = 0., 0.
        else:
            x0, y0 = x[i - 1], y[i - 1]
        x[i] = x0 + sp * np.cos(direc[i]) * dt
        y[i] = y0 + sp * np.sin(direc[i]) * dt

    return t, x, y, direc


@njit(parallel=True, fastmath=True, nogil=True)
def traj(
    dt: float,
    tmax: int,
    sp: float = settings.speed,
    init_dir: float = None,
    dphi: float = 0.5,
    bound: tuple = None,
    sq_bound: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Samples a random walk trajectory. If 'bound' is not 'None', the random walk
    is bounded by either a square if 'sq_bound' is True, or a circular boundary
    otherwise.

    Parameters
    ----------
        dt (float):         time step in seconds
        tmax (int):         total simulation time in seconds
        sp (float):         speed of agent in centimeters per second
        init_dir (float):   initial direction in radians
        dphi (float):       the tortuosity of the trajectory
        bound (float):      environment size in cm
        sq_bound (bool):    if True, boundary is a square, else a circle

    Returns
    ----------
        t:          vector of times
        x:          vector of x coordinates
        y:          vector of y coordinates
        direc:      vector of movement directions
    """
    # initialise time and trajectory
    t = np.arange(0, tmax + dt, dt)
    x, y, direc = np.zeros((3, len(t)))

    # random initial movement direction
    if init_dir is None:
        olddir = np.random.uniform(- np.pi, np.pi)
    else:
        olddir = init_dir

    # sample trajectory from random walk
    for i in range(1, len(t)):
        sigma_theta = dphi  # initialize sigma_theta. Increases if agent stuck
        # we will use length of step to check if agent stepped outside of the
        # boundary
        length = 0
        count = 0
        while np.round(length, 5) != np.round(sp * dt, 5):
            newdir = olddir \
                    + sigma_theta * np.sqrt(dt) * np.random.randn()
            dx = sp * np.cos(newdir) * dt
            dy = sp * np.sin(newdir) * dt

            # enforce boundary if a boundary function is used
            if bound is not None:
                # get amount agent will overstep a boundary, then
                # subtract that amount from the change in position
                if not sq_bound:
                    x_over, y_over, _ = circle_bound(
                        x[i - 1] + dx, y[i - 1] + dy, bound
                    )
                else:
                    x_over, y_over, _ = square_bound(
                        x[i - 1] + dx, y[i - 1] + dy, bound
                    )

                dx = dx - x_over
                dy = dy - y_over

            length = np.sqrt(dx**2 + dy**2)
            count += 1  # tracking this loop helps find if agent is stuck
            # if agent is stuck, iteratively increase tortuosity
            if count == 50:
                sigma_theta *= 1.1
                count = 0

        # calculate new position
        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy

        # update movement direction
        olddir = np.arctan2(dy, dx)
        direc[i] = olddir

    return t, x, y, direc


@jit
def traj_star2(
    phbins: int,
    rmax: float,
    dt: float,
    sp: float = settings.speed,
    offset: list = [0., 0.],
    grsc: float = settings.grsc
) -> Tuple[float, float, float, float]:
    """
    Samples a star-like walk trajectory

    Parameters
    ----------
        phbins (int):           number of angles to sample from [0, 2*pi)
        rmax (float):           length of a linear segment in centimeters
        dt (float):             time step size in seconds
        sp (float):             speed of agent in centimeters per second

    Returns
    ----------
        t (np.ndarray):         vector of times
        x (np.ndarray):         vector of x coordinates
        y (np.ndarray):         vector of y coordinates
        direc (np.ndarray):     vector of movement directions
    """
    nlin = int(rmax / sp / dt)
    tmax = rmax/sp * phbins
    t = np.linspace(0, tmax, int(tmax / dt))

    oxr, oyr = convert_to_rhombus(offset[0], offset[1])

    r, phi = np.meshgrid(
        np.linspace(0, rmax, nlin),
        np.linspace(-np.pi, np.pi, settings.phbins, endpoint=False)
    )
    X, Y = r*np.cos(phi), r*np.sin(phi)

    idx = np.random.permutation(settings.phbins)

    x = X[idx, :].flatten() + oxr
    y = Y[idx, :].flatten() + oyr
    direc = phi[idx, :].flatten()
    assert len(direc) == len(t)

    return t, x, y, direc


def pwliner(dt2, t, x, y, direc):
    """
    Resamples a given trajectory by linearly interpolating between points and
    using a new time step size

    Args:
        dt2 (_type_): _description_
        t (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        direc (_type_): _description_

    Returns:
        _type_: _description_
    """
    dt = np.mean(np.diff(t))
    t2 = np.arange(0, t[-1] + dt2, dt2)
    fx = interpolate.interp1d(t, x)
    fy = interpolate.interp1d(t, y)
    x2 = fx(t2)
    y2 = fy(t2)
    direc2 = np.concatenate((np.zeros(1), np.repeat(direc[1:], int(dt/dt2))))

    return t2, x2, y2, direc2


def load_traj() -> None:
    """
    Loads all available experimental subject path data from the data/paths
    directory

    Returns:
        trajec (dict): dictionary of trajectories where the keys
                        are the subject
    """
    temp_th, temp_of = load_data()
    temp_th = clean_data(temp_th)
    temp_of = clean_data(temp_of)
    trajec_th = get_dirs(temp_th)
    trajec_of = get_dirs(temp_of)

    return trajec_th, trajec_of


def rotate_traj(
    trajec: np.ndarray,
    theta: float = np.pi / 2
) -> None:
    """
    Rotates a trajectory by theta radians

    Args:
        trajec (tuple): trajectory to be rotated
        theta (float): angle in radians to rotate by

    Returns:
        trajec_rot (tuple): rotated trajectory
    """
    t, x, y, direc = trajec
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]]
    )
    # rotate trajectory
    xy = np.array([x, y])
    xy_rot = rot @ xy
    x_rot = xy_rot[0, :]
    y_rot = xy_rot[1, :]
    direc_rot = direc + theta
    direc_rot = (direc_rot + np.pi) % (2 * np.pi) - np.pi

    return t, x_rot, y_rot, direc_rot


def gridpop(
    N: int,
    grsc: float,
    phbins: int,
    traj: np.ndarray,
    ox: list,
    oy: list,
    ang: float = 0.,
    amax: float = 1,
    propconj: float = 0.33,
    kappa: float = 50,
    jitter: float = 10,
    shufforient: bool = False,
    conj: bool = False,
    repsuppcell: bool = False,
    repsupppop: bool = False,
    tau_rep: float = 3.5,
    w_rep: float = 50,
    mus: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [summary]

    Args:
        phbins (int): number of bins to partition the space of angles for
            the mean firing rates
        name (str): name of the corresponding trajectory
        traj (np.ndarray): the trajectory to use when calculating grid
            cell activity
        Amax (float, optional): maximum firing rate. Defaults to 20.
        propconj (float, optional): proportion of conjunctive cells.
            Defaults to 0.66.
        kappa (float, optional): [description]. Defaults to 50.
        jitter (float, optional): [description]. Defaults to 10.
        shufforient (bool, optional): [description]. Defaults to False.
        conj (bool, optional): [description]. Defaults to False.
        repsuppcell (bool, optional): Whether to use the repetition
            suppression model for grid cell dynamics. Defaults to False.
        repsupppop (bool, optional): [description]. Defaults to False.
        tau_rep (float, optional): suppression time constant.
            Defaults to 3.5.
        w_rep (float, optional): weight of suppression. Defaults to 50.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple of binned running directions, mean firing rate, and
            firing rate
    """
    t, x, y, direc = traj
    N = len(ox)
    oxr, oyr = convert_to_rhombus(ox, oy)
    fr = np.zeros((N, len(x)))

    print("\nCalculating grid cell population activity...")
    for n in tqdm(range(N)):
        if shufforient:
            ang = 360*np.random.rand()
            if repsuppcell:
                fr[n, :] += adap_euler(
                    amax*grid(
                        x,
                        y,
                        grsc=grsc,
                        angle=ang,
                        offs=(oxr[n], oyr[n])
                    ),
                    t,
                    tau_rep,
                    w_rep
                )
            elif conj:
                if n <= int(propconj * N):
                    if mus is None:
                        mu = np.mod(ang + jitter * np.random.randn(), 360)
                    else:
                        mu = mus[n]
                    fr[n, :] += amax * grid(
                        x,
                        y,
                        grsc=grsc,
                        angle=ang,
                        offs=(oxr[n], oyr[n])
                    ) * np.exp(
                        kappa * np.cos(direc - np.pi / 180. * mu)
                    ) / (sc.i0(kappa))
                else:
                    fr[n, :] += amax * grid(
                        x,
                        y,
                        grsc=grsc,
                        angle=ang,
                        offs=(oxr[n], oyr[n])
                    )
            else:
                fr[n, :] += amax * grid(
                    x, y, grsc=grsc, angle=ang, offs=(oxr[n], oyr[n])
                )
        else:
            if repsuppcell:
                fr[n, :] += adap_euler(
                    amax*grid(
                        x,
                        y,
                        grsc=grsc,
                        angle=ang,
                        offs=(oxr[n], oyr[n])
                    ),
                    t,
                    tau_rep,
                    w_rep
                )
            elif conj:
                if n <= int(propconj * N):
                    if mus is None:
                        mu = np.mod(
                            np.random.randint(0, 6, 1) *
                            60 + jitter * np.random.randn(),
                            360
                        )
                    else:
                        mu = mus[n]
                    fr[n, :] += amax * \
                        grid(
                            x,
                            y,
                            grsc=grsc,
                            offs=(oxr[n], oyr[n])) * np.exp(
                        kappa * np.cos(direc - np.pi / 180. * mu)) / (
                                    sc.i0(kappa))
                else:
                    fr[n, :] += amax * grid(
                        x, y, grsc=grsc, offs=(oxr[n], oyr[n])
                    )
            else:
                fr[n, :] += amax*grid(
                    x,
                    y,
                    grsc=grsc,
                    angle=0,
                    offs=(oxr[n], oyr[n])
                )

    summed_fr = np.sum(fr, axis=0)
    if repsupppop:
        summed_fr = adap_euler(summed_fr, t, tau_rep, w_rep)

    space = np.mean(np.diff(np.linspace(-np.pi, np.pi, phbins+1)))
    direc_binned = np.linspace(
        -np.pi, np.pi, phbins + 1
    ).astype(np.float32)[:-1]
    bin_edges = np.linspace(
        -np.pi - space/2, np.pi + space/2, phbins + 2
    ).astype(np.float32)
    fr_mean = np.zeros(phbins)
    for id in range(len(direc_binned)):
        ind = (direc >= bin_edges[id]) * (direc < bin_edges[id + 1])
        if summed_fr[ind].size > 0:
            fr_mean[id] = np.sum(summed_fr[ind]) / \
                len(summed_fr[ind])

    return direc_binned, fr_mean, fr, summed_fr


@njit(parallel=True, fastmath=True, nogil=True)
def gridpop_clustering(
    N: int,
    grsc: float,
    phbins: int,
    traj: np.ndarray,
    oxr: list,
    oyr: list,
    ang: float = 0.,
    amax: float = settings.amax,
    shufforient: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates grid cell population activity for the clustering hypothesis

    Args:
        N (int): number of cells
        grsc (float): gridscale
        phbins (int): number of bins to partition the space of angles
        traj (np.ndarray): the trajectory to use
        oxr (list): x-offsets of the cells
        oyr (list): y-offsets of the cells
        ang (float, optional): angular phase of cell grid fields.
            Defaults to 0.
        Amax (float, optional): maximum firing rate divided by 8.
            Defaults to 1.
        shufforient (bool, optional): whether to shuffle the angular phases of
            the cells. Defaults to False.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple of binned running directions (M time steps), mean firing
            rate (phbins movement directions), time dependent firing rate
            (N cells x M time steps) and summed firing rate (M time steps)

    """
    t, x, y, direc = traj
    N = len(oxr)
    # oxr, oyr = convert_to_rhombus(ox, oy)
    fr = np.zeros((N, len(x)))

    for n in np.arange(N):
        ang = 360*np.random.rand() if shufforient else 0
        fr[n, :] = amax * grid(
            x,
            y,
            grsc=grsc,
            angle=ang,
            offs=(oxr[n], oyr[n])
        )

    summed_fr = np.sum(fr, axis=0)

    space = np.mean(np.diff(np.linspace(-np.pi, np.pi, phbins+1)))
    direc_binned = np.linspace(
        -np.pi, np.pi, phbins + 1
    ).astype(np.float32)[:-1]
    bin_edges = np.linspace(
        -np.pi - space/2, np.pi + space/2, phbins + 2
    ).astype(np.float32)
    fr_mean = np.zeros(phbins)
    for id in range(len(direc_binned)):
        ind = (direc >= bin_edges[id]) * (direc < bin_edges[id + 1])
        if summed_fr[ind].size > 0:
            fr_mean[id] = np.sum(summed_fr[ind]) / \
                len(summed_fr[ind])

    return direc_binned, fr_mean, fr, summed_fr


@njit(parallel=True, fastmath=True, nogil=True)
def gridpop_repsupp(
    N: int,
    grsc: float,
    phbins: int,
    traj: np.ndarray,
    oxr: list,
    oyr: list,
    ang: float = 0.,
    amax: float = settings.amax,
    shufforient: bool = False,
    repsuppcell: bool = True,
    repsupppop: bool = False,
    tau_rep: float = settings.tau_rep,
    w_rep: float = settings.w_rep
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [summary]

    Args:
        phbins (int): number of bins to partition the space of angles for
            the mean firing rates
        name (str): name of the corresponding trajectory
        traj (np.ndarray): the trajectory to use when calculating grid
            cell activity
        Amax (float, optional): maximum firing rate. Defaults to 20.
        propconj (float, optional): proportion of conjunctive cells.
            Defaults to 0.66.
        kappa (float, optional): [description]. Defaults to 50.
        jitter (float, optional): [description]. Defaults to 10.
        shufforient (bool, optional): [description]. Defaults to False.
        conj (bool, optional): [description]. Defaults to False.
        repsuppcell (bool, optional): Whether to use the repetition
            suppression model for grid cell dynamics. Defaults to False.
        repsupppop (bool, optional): [description]. Defaults to False.
        tau_rep (float, optional): suppression time constant.
            Defaults to 3.5.
        w_rep (float, optional): weight of suppression. Defaults to 50.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple of binned running directions (M time steps), mean firing
            rate (phbins movement directions), time dependent firing rate
            (N cells x M time steps) and summed firing rate (M time steps)
    """
    t, x, y, direc = traj
    N = len(oxr)
    fr = np.zeros((N, len(x)))

    for n in range(N):
        ang = 360*np.random.rand() if shufforient else 0
        if repsuppcell:
            fr[n, :] = adap_euler(
                s=amax*grid(
                    x,
                    y,
                    grsc=grsc,
                    angle=ang,
                    offs=(oxr[n], oyr[n])
                ),
                tt=t,
                tau=tau_rep,
                w=w_rep
            )
        else:
            fr[n, :] = amax * grid(
                x,
                y,
                grsc=grsc,
                angle=ang,
                offs=(oxr[n], oyr[n])
            )

    summed_fr = np.sum(fr, axis=0)
    if repsupppop:
        summed_fr = adap_euler(summed_fr, t, tau_rep, w_rep)

    space = np.mean(np.diff(np.linspace(-np.pi, np.pi, phbins+1)))
    direc_binned = np.linspace(
        -np.pi, np.pi, phbins + 1
    ).astype(np.float32)[:-1]
    bin_edges = np.linspace(
        -np.pi - space/2, np.pi + space/2, phbins + 2
    ).astype(np.float32)
    fr_mean = np.zeros(phbins)
    for id in range(len(direc_binned)):
        ind = (direc >= bin_edges[id]) * (direc < bin_edges[id + 1])
        if summed_fr[ind].size > 0:
            fr_mean[id] = np.sum(summed_fr[ind]) / \
                len(summed_fr[ind])

    return direc_binned, fr_mean, fr, summed_fr


def gridpop_conj(
    N: int,
    grsc: float,
    phbins: int,
    traj: np.ndarray,
    oxr: list,
    oyr: list,
    ang: float = 0.,
    amax: float = settings.amax,
    propconj: float = settings.propconj_i,
    kappa: float = settings.kappac_i,
    jitter: float = settings.jitterc_i,
    shufforient: bool = False,
    mus: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates the activity of 'N' grid cells using the trajectory 'traj' under
    the conjunctive grid by head-direction cell hypothesis.

    Args:
        phbins (int): number of bins to partition the space of angles for
            the mean firing rates
        name (str): name of the corresponding trajectory
        traj (np.ndarray): the trajectory to use when calculating grid
            cell activity
        Amax (float, optional): maximum firing rate. Defaults to 20.
        propconj (float, optional): proportion of conjunctive cells.
            Defaults to 0.66.
        kappa (float, optional): [description]. Defaults to 50.
        jitter (float, optional): [description]. Defaults to 10.
        shufforient (bool, optional): [description]. Defaults to False.
        conj (bool, optional): [description]. Defaults to False.
        repsuppcell (bool, optional): Whether to use the repetition
            suppression model for grid cell dynamics. Defaults to False.
        repsupppop (bool, optional): [description]. Defaults to False.
        tau_rep (float, optional): suppression time constant.
            Defaults to 3.5.
        w_rep (float, optional): weight of suppression. Defaults to 50.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple of binned running directions (M time steps), mean firing
            rate (phbins movement directions), time dependent firing rate
            (N cells x M time steps) and summed firing rate (M time steps)
    """
    t, x, y, direc = traj
    N = len(oxr)
    fr = np.zeros((N, len(x)))

    for n in range(N):
        ang = 2*np.pi*np.random.rand() if shufforient else ang
        if n <= int(propconj * N):
            if mus is None:
                # mu = np.mod(ang + jitter * np.random.randn(), 360)
                mu = np.mod(
                    np.random.randint(0, 6, 1) *
                    60 + jitter * np.random.randn(),
                    360
                )
            else:
                mu = mus[n]
            fr[n, :] = amax * grid(
                x,
                y,
                grsc=grsc,
                angle=ang,
                offs=(oxr[n], oyr[n])
            ) * np.exp(
                kappa * np.cos(direc - np.pi / 180. * mu)
            ) / (sc.i0(kappa))
        else:
            fr[n, :] = amax * grid(
                x,
                y,
                grsc=grsc,
                angle=ang,
                offs=(oxr[n], oyr[n])
            )

    summed_fr = np.sum(fr, axis=0)

    space = np.mean(np.diff(np.linspace(-np.pi, np.pi, phbins+1)))
    direc_binned = np.linspace(
        -np.pi, np.pi, phbins + 1
    ).astype(np.float32)[:-1]
    bin_edges = np.linspace(
        -np.pi - space/2, np.pi + space/2, phbins + 2
    ).astype(np.float32)
    fr_mean = np.zeros(phbins)
    for id in range(len(direc_binned)):
        ind = (direc >= bin_edges[id]) * (direc < bin_edges[id + 1])
        if summed_fr[ind].size > 0:
            fr_mean[id] = np.sum(summed_fr[ind]) / \
                len(summed_fr[ind])

    return direc_binned, fr_mean, fr, summed_fr


@njit(parallel=True, fastmath=True, nogil=True)
def gridpop_const(
    N: int,
    phbins: int,
    traj: np.ndarray,
    amax: float = settings.amax
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates grid cell population activity for the case where the grid
    function is a constant.

    Args:
        N (int): number of cells
        phbins (int): number of bins to partition the space of angles
        traj (np.ndarray): the trajectory to use
        amax (float, optional): the constant firing rate

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple of binned running directions (M time steps), mean firing
            rate (phbins movement directions), time dependent firing rate
            (N cells x M time steps) and summed firing rate (M time steps)

    """
    t, x, y, direc = traj
    fr = np.zeros((N, len(x)))

    for n in np.arange(N):
        fr[n, :] = amax

    summed_fr = np.sum(fr, axis=0)

    space = np.mean(np.diff(np.linspace(-np.pi, np.pi, phbins+1)))
    direc_binned = np.linspace(
        -np.pi, np.pi, phbins + 1
    ).astype(np.float32)[:-1]
    bin_edges = np.linspace(
        -np.pi - space/2, np.pi + space/2, phbins + 2
    ).astype(np.float32)
    fr_mean = np.zeros(phbins)
    for id in range(len(direc_binned)):
        ind = (direc >= bin_edges[id]) * (direc < bin_edges[id + 1])
        if summed_fr[ind].size > 0:
            fr_mean[id] = np.sum(summed_fr[ind]) / \
                len(summed_fr[ind])

    return direc_binned, fr_mean, fr, summed_fr


def meanfr_repsupp(
    rmax = 300,
    bins = 500,
    phbins = settings.phbins,
    N = settings.N,
    tau_rep = settings.tau_rep,
    w_rep = settings.w_rep,
    speed = settings.speed,
    plot = False
):
    N = int(np.sqrt(N))**2
    ox, oy = gen_offsets(N=N, kappacl=0.)
    oxr,oyr = convert_to_rhombus(ox,oy)

    offs = oxr, oyr

    # setup grids
    if N==1:
        # r,phi = np.meshgrid(np.linspace(0,rmax,bins), np.linspace(0,360,phbins))
        r,phi = np.meshgrid(np.linspace(0,rmax,bins), np.linspace(0,2 * np.pi,phbins))
        r = np.expand_dims(r, axis=2)
        phi = np.expand_dims(phi, axis=2)
        X,Y = r*np.cos(phi), r*np.sin(phi)
        grids = grid_meanfr(X, Y, offs=(oxr,oyr))
        grids2 = grids.copy()

    # how does meanfr depend on bins ... 
    # Naomi showed that the different peaks in meanfr seem to have different 
    # heights for higher number of bins??
    else:
        # r,phi,indoff = np.meshgrid(
        #     np.linspace(0,rmax,bins),
        #     np.linspace(0,360,phbins),
        #     np.arange(len(ox))
        # )
        r,phi,indoff = np.meshgrid(
            np.linspace(0,rmax,bins),
            np.linspace(0,2 * np.pi,phbins),
            np.arange(len(ox))
        )
        X,Y = r*np.cos(phi), r*np.sin(phi)

        # calculate integrated firing rate as a function of the movement angle
        grids = grid_meanfr(X,Y,offs=(oxr,oyr))
        grids2 = grids.copy()   

    if plot:
        plt.figure(figsize=(10,10))
        for i in range(int(phbins / 2)):
            plt.plot(X[2*i,:,0],Y[2*i,:,0])
        plt.savefig(
            os.path.join(
                settings.loc,
                "repsupp",
                "fig5",
                "meanfr",
                f"path.png"
            )
        )
        plt.close()

    #speed = 1. # grid scale/s
    #tau_rep = 1. #s
    #w_rep = 3.
    #s = grids[0,:,0]

    tt = np.linspace(0,rmax/speed,bins)
    #aa = np.zeros(np.shape(grids))
    for idir in range(phbins):
        for ic in range(N):
            v = adap_euler(grids[idir,:,ic],tt,tau_rep,w_rep)
            grids[idir,:,ic] = v
            #aa[idir,:,ic] = a

    if plot:
        plt.figure(figsize=(20,6))
        plt.plot(tt * speed,grids2[0,:,0],'r-')
        plt.plot(tt * speed,grids[0,:,0],'b-')
        # plt.xlim(0, np.amax(tt * speed) / 2)
        plt.xlabel("position (cm)")
        plt.ylabel("firing rate (spks/s)")
        plt.savefig(
            os.path.join(
                settings.loc,
                "repsupp",
                "fig5",
                "meanfr",
                f"grid.png"
            )
        )
        plt.close()

    if N==1:
        meanfr_base = np.sum(grids2,axis=1)/bins
        meanfr = np.sum(grids,axis=1)/bins
        meanfr = meanfr/np.mean(meanfr)*np.mean(meanfr_base)
    else:
        meanfr_base = np.sum(np.sum(grids2,axis=1)/bins,axis=1)
        meanfr = np.sum(np.sum(grids,axis=1)/bins,axis=1)
        meanfr = meanfr/np.mean(meanfr)*np.mean(meanfr_base)
    gr2 = np.sum(grids, axis=2)
    gr60 = np.abs(np.sum(gr2 * np.exp(-6j*phi[:, :, 0])))/np.size(gr2)
    gr0 = np.abs(np.sum(gr2 * np.exp(-0j*phi[:, :, 0])))/np.size(gr2)
    # gr60 = gr60 / gr0
    gr60_path = np.abs(np.sum(np.exp(-6j*phi[:, :, 0])))/(np.shape(phi)[0]*np.shape(phi)[1])

    return gr60.reshape(-1), gr60_path.reshape(-1), meanfr.reshape(-1), offs, gr2
