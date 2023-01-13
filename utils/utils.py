import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable
import os
import pickle
import utils.settings as settings
from numba import njit, jit
import matplotlib.collections as mcoll
import matplotlib.path as mpath


@njit(parallel=True, fastmath=True, nogil=True)
def grid(
    X: np.ndarray,
    Y: np.ndarray,
    grsc: int = 30,
    angle: float = 0,
    offs: np.ndarray = np.array([0., 0.])
) -> np.ndarray:
    """
    Returns the activity profile G of a grid cell

    Parameters
    -----------
        X:      grid positions along x-axis
        Y:      grid positions along y-axis
        grsc:   grid scale
        angle:  angular phase of the grid in radians
        offs:   offsets of the grid

    Returns
    -----------
        np.ndarray: grid cell activity at positions (X,Y)
    """
    # calculate activity profile for all positions
    output = np.ones((3, len(X)))
    for i in np.arange(3):
        output[i, :] = 1 + np.cos(
            4 * np.pi * np.sin(np.pi * i / 3 + angle) *
            (X - grsc * offs[0]) / (np.sqrt(3) * grsc) +
            4 * np.pi * np.cos(np.pi * i / 3 + angle) *
            (Y - grsc * offs[1]) / (np.sqrt(3) * grsc)
        )
    
    output = output[0, :] * output[1, :] * output[2, :]

    return output
    

@jit
def grid_2d(
    X: np.ndarray,
    Y: np.ndarray,
    grsc: int = 30,
    angle: float = 0,
    offs: np.ndarray = np.array([0., 0.])
) -> np.ndarray:
    """
    Returns the activity profile G of a grid cell

    Parameters
    -----------
        X:      grid positions along x-axis
        Y:      grid positions along y-axis
        grsc:   grid scale
        angle:  angular phase of the grid in radians
        offs:   offsets of the grid

    Returns
    -----------
        np.ndarray: grid cell activity at positions (X,Y)
    """
    # calculate activity profile for all positions
    output = np.ones((3, len(X), len(Y)))
    for i in np.arange(3):
        output[i, :, :] = 1 + np.cos(
            4 * np.pi * np.sin(np.pi * i / 3 + angle) *
            (X - grsc * offs[0]) / (np.sqrt(3) * grsc) +
            4 * np.pi * np.cos(np.pi * i / 3 + angle) *
            (Y - grsc * offs[1]) / (np.sqrt(3) * grsc)
        )
    output = output[0, :, :] * output[1, :, :] * output[2, :, :]

    return output


@njit(parallel=True, fastmath=True, nogil=True)
def grid_meanfr(
    X: np.ndarray,
    Y: np.ndarray,
    grsc: int = 30,
    angle: float = 0,
    offs: np.ndarray = np.array([0., 0.])
) -> np.ndarray:
    """
    Returns the activity profile G of a grid cell

    Parameters
    -----------
        X:      grid positions along x-axis
        Y:      grid positions along y-axis
        grsc:   grid scale
        angle:  angular phase of the grid in radians
        offs:   offsets of the grid

    Returns
    -----------
        np.ndarray: grid cell activity at positions (X,Y)
    """
    # calculate activity profile for all positions
    # if len(X.shape) == 2:
    #     X = np.expand_dims(X, axis=2)
    #     Y = np.expand_dims(Y, axis=2)
    assert len(X.shape) == 3 and len(Y.shape) == 3,\
        "X and Y must be 3D arrays (phbins, bins, N cells)"
    output = np.ones((3, X.shape[0], X.shape[1], X.shape[2]))
    for i in np.arange(3):
        output[i, :, :, :] = (1 + np.cos(
            4 * np.pi * np.sin(np.pi * i / 3 + angle) *
            (X - grsc * offs[0]) / (np.sqrt(3) * grsc) +
            4 * np.pi * np.cos(np.pi * i / 3 + angle) *
            (Y - grsc * offs[1]) / (np.sqrt(3) * grsc)
        ))
    output = output[0, :, :] * output[1, :, :] * output[2, :, :]

    return output


@jit
def convert_to_rhombus(x, y):
    """
    Transforms spatial coordinates to the space of the unit rhombus

    Args:
        x (float): coordinates in the x dimension
        y (float): coordinates in the y dimension

    Returns:
        (float, float): tuple of floats representing the rhombus-transformed 
            coordinates in the x and y dimension respectively.
    """
    return np.asarray(x + 0.5 * y), np.asarray(np.sqrt(3) / 2 * y)


@njit(fastmath=True, nogil=True)
def adap_euler(s, tt, tau, w):
    """Euler-solve a simple adaptation model for the repetition-suppression
    hypothesis.

    Args:
        s (_type_): _description_
        tt (_type_): _description_
        tau (_type_): _description_
        w (_type_): _description_

    Returns:
        _type_: _description_
    """
    v = np.zeros(np.shape(tt))
    a = np.zeros(np.shape(tt))
    v[0] = s[0]
    a[0] = 0
    dt = np.mean(np.diff(tt))
    for it in range(len(v)-1):
        # dv = (-w*a[it]+s[it]-v[it])/0.01*dt
        v[it+1] = max(s[it] - w*a[it], 0)
        da = (v[it]-a[it])/tau*dt
        a[it+1] = a[it] + da
    return np.where(v > 0, v, 0)  # rectify result


@njit
def get_hexsym(
    summed_fr: np.ndarray,
    traj: np.ndarray,
    fold: int = 6
) -> float:
    """
    Returns the n-fold symmetry for an array of summed firing rates and a
    trajectory.

    Args:
        summed_fr (np.ndarray): Array firing rates over time summed over N 
            cells
        traj (np.ndarray): Trajectory corresponding to firing rate array. with
            shape [time, x, y, direc]. "direc" should be in radians.
        fold (int, optional): Order of angular firing rate symmetry to measure.
            Defaults to 6.

    Returns:
        float: the n-fold symmetry.
    """
    t, x, y, direc = traj
    nsym = np.abs(
        np.sum(summed_fr * np.exp(- fold * direc * 1j)) / len(direc)
    )

    return nsym


@jit
def get_pathsym(
    traj: np.ndarray,
    fold: int = 6
) -> float:
    """
    Returns the n-fold path symmetry for a given trajectory

    Args:
        traj (np.ndarray): Trajectory array with shape [time, x, y, direc].
            "direc" should be in radians.
        fold (int, optional): Order of path symmetry to measure.
    """
    t, x, y, direc = traj
    # pathsym = np.abs(np.sum(np.exp(- fold * direc[1:] * 1j))) / (len(direc[1:]))
    pathsym = np.abs(np.sum(np.exp(- fold * direc * 1j))) / (len(direc))

    return pathsym


def get_pathsyms():
    """
    Loads and returns saved path symmetries

    Returns
    ----------
        star_path60 (float):    path hexasymmetry for a star-like run
        pwl_path60 (list):      list of path hexasymmetries for a piecewise
                                    linear run
        rw_path60 (list):       list of path hexasymmetries for a random walk
    """
    pathsyms_fname = settings.pathsyms_fname
    assert os.path.isfile(pathsyms_fname),\
        "No pathsyms file! Run produce_pathsyms.py"
    with open(pathsyms_fname, 'rb') as f:
        pathsyms = pickle.load(f)
    
    return pathsyms["star"], pathsyms["pwl"], pathsyms["rw"]


def get_hexsym_binning(meanfr, phbins=360):
    ntile = 4
    fr2 = np.tile(meanfr,ntile) - np.mean(meanfr) # tile to account for periodicity
    power = 2./len(fr2) * abs(np.fft.fft(fr2))[:len(fr2)//2]
    freq = np.fft.fftfreq(fr2.size, d=1/phbins)[:len(fr2)//2]
    ff = np.linspace(0,1./2., ntile*phbins//2)
    
    if 0:
        plt.figure()
        plt.plot(ff,power)
        plt.xticks([1/360,1/90,1/60,1/45,1/30,1/17,1/15], ['360','90','60','45','30','17','15'])
    
    gr60 = power[np.argmin(abs(freq-1./60*360))] # pick index which is closest to 60deg
    # gr45 = power[np.argmin(abs(ff-1./45*360/phbins))]
    # gr90 = power[np.argmin(abs(ff-1./90*360/phbins))]
    # sym5 = power[np.argmin(abs(ff-1./(360/5)*360/phbins))]
    # sym7 = power[np.argmin(abs(ff-1./(360/7)*360/phbins))]
    # gr17 = power[np.argmin(abs(ff-1./17*360/phbins))]
    
    # ph = np.angle(np.fft.fft(fr2))[:len(fr2)//2]
    # p60 = ph[np.argmin(abs(ff-1./60*360/phbins))]/2/np.pi*60
    # p45 = ph[np.argmin(abs(ff-1./45*360/phbins))]/2/np.pi*45
    # p90 = ph[np.argmin(abs(ff-1./90*360/phbins))]/2/np.pi*90
        
    return gr60,power,freq


def get_hexsym_binning2(meanfr):
    ft = np.fft.fft(meanfr)
    frq = np.fft.fftfreq(len(meanfr))
    frq_deg = frq * 360
    # Find the index of the frequency 6 component
    idx = np.argmin(np.abs(frq_deg - 6.))

    # Compute the power of the frequency 6 component
    power = np.abs(ft[idx])**2

    # Compute the sum of the powers of all frequency components
    total_power = np.sum(np.abs(ft)**2)

    # Compute the normalized power of the frequency 6 component
    norm_power = power / total_power

    return norm_power, frq


def visualise_traj(
    traj: tuple,
    fname: str = "None",
    grsc: float = 30.,
    angle: float = 0.,
    offs: list = None,
    xticks: list = None,
    yticks: list = None,
    extent: float = None,
    title: str = None
):
    """
    Plots and saves a given trajectory for a realisation of a grid field

    Args:
        traj (np.ndarray): _description_
        save (str, optional): _description_. Defaults to "None".
        folder (str, optional): _description_. Defaults to "None".
        grsc (float, optional): _description_. Defaults to 30..
        angle (float, optional): _description_. Defaults to 0..
        offs (list, optional): _description_. Defaults to None.
    """
    if not os.path.exists(fname):
        t, x, y, direc = traj

        # plot a grid field
        nx, ny = (4000, 4000)
        # xmin = int(np.amin(x) * 1.2)
        # ymin = int(np.amin(y) * 1.2)
        xmax = int(np.amax(np.abs(x)) * 1.2)
        ymax = int(np.amax(np.abs(y)) * 1.2)
        rmax = max(xmax, ymax)
        max_bound = 6000
        rmax = min(rmax, max_bound) if not extent else extent
        xg = np.linspace(-rmax, rmax, nx)
        yg = np.linspace(-rmax, rmax, ny)
        xx, yy = np.meshgrid(xg, yg)
        plt.ioff()
        plt.figure(figsize=(6, 6))
        plt.rcParams.update({'font.size': settings.fs})
        if offs is not None:
            plt.imshow(
                grid_2d(xx, yy, grsc=grsc, offs=offs),
                aspect="auto",
                extent=(-rmax, rmax, -rmax, rmax)
            )
        plt.plot(x, y, lw=2, color="red")
        if xticks:
            plt.xticks(np.array(xticks))
        if yticks:
            plt.yticks(np.array(yticks))
        if title:
            plt.rcParams['axes.titley'] = 1.0
            plt.rcParams['axes.titlepad'] = 10.0
            plt.title(title)
        # plt.xlabel("x (cm)")
        # plt.ylabel("y (cm)")
        # plt.tight_layout()
        # plt.show()
        if fname is not None:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(fname)
            plt.close()
    else:
        print(f"trajectory plot exists, skipping")


def get_plotting_grid(
    trajec,
    offs,
    nx = settings.nx,
    ny = settings.ny,
    grsc = settings.grsc,
    extent: float = None,
    max_bound: float = 6000,
    xticks: list = [],
    yticks: list = [],
    title: str = None,
    titlepos: list = None,
    titlerot: float = 0.,
    trajcolor: str = "red",
    trajprop: float = 1,
    ylabel: str = None,
    xlabel: str = None
):
    t, x, y, direc = trajec
    x, y = x[:int(len(x)*trajprop)], y[:int(len(y)*trajprop)]
    xmax = int(np.amax(np.abs(x)) * 1.2)
    ymax = int(np.amax(np.abs(y)) * 1.2)
    rmax = max(xmax, ymax)
    rmax = min(rmax, max_bound) if not extent else extent
    xg = np.linspace(-rmax, rmax, nx)
    yg = np.linspace(-rmax, rmax, ny)
    xx, yy = np.meshgrid(xg, yg)
    # grid = grid_2d(xx, yy, grsc=grsc, offs=offs)

    plt.imshow(
        grid_2d(xx, yy, grsc=grsc, offs=offs),
        aspect="equal",
        extent=(-rmax, rmax, -rmax, rmax)
    )
    plt.plot(x, y, lw=2, color=trajcolor)
    plt.xticks(np.array(xticks))
    plt.yticks(np.array(yticks))
    plt.xlim(-extent, extent)
    plt.ylim(-extent, extent)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.rcParams['axes.titley'] = 1.0
        plt.rcParams['axes.titlepad'] = 10.0
        if titlepos:
            plt.title(title, x=titlepos[0], y=titlepos[1], rotation=titlerot)
        else:
            plt.title(title, rotation=titlerot)

    # return plot


def ax_pos(
    ax, 
    movex,
    movey,
    scalex,
    scaley
):
    """
    Moves or scales an axis of a matplotlib figure

    Args:
        ax (axes): instance of a matplotlib axes
        movex (float): distance to move in x direction
        movey (float): distance to move in y direction
        scalex (float): scaling factor in x direction
        scaley (float): scaling factor in y direction
    """
    # get current axes coordinates
    posax = ax.get_position()
    pointsax = posax.get_points()

    # centering for scaling operation
    mean_pointsax = np.array(
        [(pointsax[0][0] + pointsax[1][0])/2,
        (pointsax[0][1] + pointsax[1][1])/2]
    )

    pointsax -= mean_pointsax

    # perform scaling operation
    pointsax[0][0] *= scalex
    pointsax[0][1] *= scaley
    pointsax[1][0] *= scalex
    pointsax[1][1] *= scaley

    # perform translation operation
    pointsax[0][0] += movex
    pointsax[0][1] += movey
    pointsax[1][0] += movex
    pointsax[1][1] += movey

    # return to original coordinate system
    pointsax += mean_pointsax

    # set new coordinates
    posax.set_points(pointsax)
    ax.set_position(posax)


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def get_t6upperbnd(tort, dt, m):
    return np.sqrt((1 + 2 / (np.exp(18 * tort**2 * dt) - 1)) / m)