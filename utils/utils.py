import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import settings
import matplotlib.collections as mcoll
from numba import njit, jit
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from math import remainder


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
    assert len(X.shape) == 3 and len(Y.shape) == 3, \
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
    assert not np.isnan(tt).any(), "tt contains NaN values"
    v = np.zeros(np.shape(tt))
    a = np.zeros(np.shape(tt))
    v[0] = s[0]
    a[0] = 0
    dt = 0
    if len(tt) > 1:
        dt = (tt[-1] - tt[0]) / (len(tt) - 1)
    for it in range(len(v)-1):
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
    assert os.path.isfile(pathsyms_fname), \
        "No pathsyms file! Run produce_pathsyms.py"
    with open(pathsyms_fname, 'rb') as f:
        pathsyms = pickle.load(f)

    return pathsyms["star"], pathsyms["pwl"], pathsyms["rw"]


def divide_dataset(x, y, num_groups):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if num_groups < 1:
        raise ValueError("num_groups must be greater than 0.")

    # Calculate the size of each group
    group_size = len(x) // num_groups

    # Calculate the number of elements that will be left after dividing into
    # equal groups
    remainder = len(x) % num_groups

    groups = []

    start_idx = 0
    for i in range(num_groups):
        end_idx = start_idx + group_size

        # Distribute the remainder elements one by one to the groups
        if remainder > 0:
            end_idx += 1
            remainder -= 1

        # Get the x and y values for this group
        group_x = x[start_idx:end_idx]
        group_y = y[start_idx:end_idx]

        groups.append((group_x, group_y))

        start_idx = end_idx

    return groups


def cosine_sine_regressors(x, n0=6, n_terms=1):
    """
    Generate cosine and sine regressors for the given data x

    Parameters:
    x (array): Input data
    n_terms (int): Number of terms for cosine and sine (default is 1)

    Returns:
    tuple: A tuple containing the cosine and sine regressors (arrays)
    """
    x = np.array(x)
    cosine_regressors = np.zeros((len(x), n_terms))
    sine_regressors = np.zeros((len(x), n_terms))

    for i in range(n_terms):
        cosine_regressors[:, i] = np.cos((i+n0) * x)
        sine_regressors[:, i] = np.sin((i+n0) * x)

    return cosine_regressors, sine_regressors


def get_glm_hex(x, signal):
    # Divide the dataset
    groups = divide_dataset(x, signal, num_groups=2)
    x_odd, signal_odd = groups[0]
    x_even, signal_even = groups[1]

    # Get orientation with first half of the data
    cosine_regressors, sine_regressors = cosine_sine_regressors(x_odd)
    X1 = np.hstack((cosine_regressors, sine_regressors))
    model1 = LinearRegression()
    model1.fit(X1, signal_odd)

    # Get the beta values of the GLM
    betas = model1.coef_
    angle = np.arctan2(betas[1], betas[0]) / 6
    # angle = np.arctan(-betas[0] / betas[1]) / 6

    # Get hexasymmetry with the second half
    X2 = np.cos(6 * (x_even + angle))
    X2 = X2.reshape(-1, 1)
    model2 = LinearRegression()
    model2.fit(X2, signal_even)
    beta = model2.coef_

    hex = beta

    return hex, angle, beta, model2.intercept_


def get_glmbinning_hex_means(
    x: np.ndarray,
    signal: np.ndarray,
    bin_shift: float = np.deg2rad(15.),
    num_bins: int = 12
):
    """
    Obtain the hexasymmetry and other parameters using the Kunz method on a
    given signal that is divided into two datasets.

    Parameters
    ----------
    x : ndarray
        The vector of headings.

    signal : ndarray
        The signal to analyze.

    num_bins : int, optional
        The number of bins to use for the computation, by default 12 to match
        experimental studies.

    bin_shift : float, optional
        The shift for the bins in radians, by default np.deg2rad(15.) so that
        the center of bins is aligned to either a peak or a trough.

    real : bool, optional
        If True, the function uses a Generalized Linear Model (GLM) on the
        second half of the dataset.
        If False, the function calculates the mean of highs and lows of the
        subsampled signal, by default True.

    Returns
    -------
    tuple
        A tuple containing the hexasymmetry, amplitude, orientation,and the
        intercept from the GLMs.

    """
    # Divide dataset
    groups = divide_dataset(x, signal, num_groups=2)
    x_odd, signal_odd = groups[0]
    x_even, signal_even = groups[1]

    # Get orientation with first half of data
    cosine_regressors, sine_regressors = cosine_sine_regressors(x_odd)
    X = np.hstack((cosine_regressors, sine_regressors))
    # y_pred = model.predict(X)

    # Get the beta values of the GLM
    model = LinearRegression()
    model.fit(X, signal_odd)
    betas = model.coef_
    angle = np.arctan(betas[1] / betas[0]) / 6  # orientation

    # Get hexasymmetry with second half of data
    # Define bin edges based on orientation
    bin_edges = np.linspace(
        0. + angle - bin_shift,
        2*np.pi + angle - bin_shift,
        num_bins + 1,
        endpoint=True
    )

    # Extend heading and signal vectors for headings +- 2 pi
    ext_x = np.tile(x_even, 3)
    ext_x[:int(len(ext_x)/3)] -= 2 * np.pi
    ext_x[2*int(len(ext_x)/3):] += 2 * np.pi
    # ext_x = np.linspace(
    #     -2 * np.pi, 2 * 2 * np.pi, 3*len(x), endpoint=False
    # )
    ext_signal = np.tile(signal_even, 3)

    # Only select points between the defined bin edges
    ext_signal = ext_signal[
        np.argwhere(
            np.logical_and(ext_x > bin_edges[0], ext_x < bin_edges[-1])
        ).reshape(-1)
    ]

    # Compute the indices of the bins for the extended time and signal
    bin_indices = np.digitize(ext_x, bin_edges)
    bin_indices = bin_indices[
        np.argwhere(
            np.logical_and(ext_x > bin_edges[0], ext_x < bin_edges[-1])
        ).reshape(-1)
    ]

    # ext_x = ext_x[
    #     np.argwhere(
    #         np.logical_and(ext_x > bin_edges[0], ext_x < bin_edges[-1])
    #     ).reshape(-1)
    # ]

    # Compute the subsampled signal by averaging the signal over each bin
    subsampled_signal = []
    for i in range(1, num_bins+1):
        bin_data = ext_signal[bin_indices == i]
        bin_data = bin_data[~np.isnan(bin_data)]
        if len(bin_data) == 0:
            subsampled_signal.append(np.nan)
        else:
            subsampled_signal.append(bin_data.mean())
    subsampled_signal = np.array(subsampled_signal)

    # Flatten the subsampled signal
    subsampled_signal = np.reshape(subsampled_signal, -1)

    # Estimate amplitude and get hexasymmetry
    if True:
        # GLM on second half
        X2 = np.ones(len(subsampled_signal)).reshape(-1, 1)
        X2[1::2, :] *= -1
        model2 = LinearRegression()
        model2.fit(X2, subsampled_signal)
        amp = model2.coef_

    else:
        evens = subsampled_signal[0::2]
        evens = evens[~np.isnan(evens)]
        odds = subsampled_signal[1::2]
        odds = odds[~np.isnan(odds)]
        # If difference of means is used, amp needs to be divided by 4
        amp = (np.mean(evens) - np.mean(odds)) / 2

    hex = np.abs(amp / 2)

    return hex, amp, angle, model.intercept_


def get_aligned_idx(
    array, preferred_angle, tolerance, angperiod=np.deg2rad(60.)
):
    '''
    This function takes an array of angles in radians, a preferred angle in
    radians, and a tolerance in radians. It returns the indices of all
    directions in the array which are within the tolerance of the preferred
    angle and all its multiples.
    '''
    # Normalize array of angles to be within -pi to pi
    array = np.arctan2(np.sin(array), np.cos(array))

    # Initialise array of indices
    within_tolerance = []

    # Iterate over each angle in the array
    for i, angle in enumerate(array):
        # Calculate the difference between the movement direction and the
        # closest preferred movement direction
        diff = np.abs(remainder(angle - preferred_angle, angperiod))
        # Check if the minimum difference is within the specified tolerance
        if diff < tolerance:
            # If it is, add the index of the angle to the list
            within_tolerance.append(i)

    return within_tolerance


def get_glmbinning_hex(
    x: np.ndarray,
    signal: np.ndarray,
    bin_shift: float = np.deg2rad(15.)
):
    """
    Obtain the hexasymmetry and other parameters using the GLM method with
    binning for a given signal that is divided into two datasets.

    Parameters
    ----------
    x : ndarray
        The vector of headings.

    signal : ndarray
        The signal to analyze.

    bin_shift : float, optional
        The shift for the bins in radians, by default np.deg2rad(15.) so that
        the center of bins is aligned to either a peak or a trough.

    Returns
    -------
    tuple
        A tuple containing the hexasymmetry, amplitude, orientation,and the
        intercept from the GLMs.

    """
    # Divide dataset
    groups = divide_dataset(x, signal, num_groups=2)
    x_odd, signal_odd = groups[0]
    x_even, signal_even = groups[1]

    # Get orientation with first half of data
    cosine_regressors, sine_regressors = cosine_sine_regressors(x_odd)
    X = np.hstack((cosine_regressors, sine_regressors))

    # Get the beta values of the GLM
    model = LinearRegression()
    model.fit(X, signal_odd)
    betas = model.coef_
    angle = np.arctan2(betas[1], betas[0]) / 6
    # angle = (np.arctan(- betas[0] / betas[1])) / 6 + np.pi

    # GLM on second half
    align_idx = get_aligned_idx(x_even, angle, bin_shift)
    misalign_idx = np.delete(np.arange(len(x_even)), align_idx)
    X2 = np.ones(len(x_even)).reshape(-1, 1)
    X2[misalign_idx] *= -1
    model2 = LinearRegression()
    model2.fit(X2, signal_even)
    amp = model2.coef_

    # Hexasymmetry is half the of the beta value
    hex = amp

    return hex, amp, angle, model2.intercept_


def _corr(x, y, axis=0):
    return np.sum((x - x.mean(axis=axis, keepdims=True)) *
                  (y - y.mean(axis=axis, keepdims=True)), axis=axis) \
            / np.std(x, axis=axis) / np.std(y, axis=axis) / x.shape[axis]


def circlin_cl(linear_var, circular_var, radians=True, fold=6.):
    # Convert the circular variable to radians if it's in degrees
    if not radians:
        circular_var = np.radians(circular_var)

    # Compute the cosine and sine of the circular variable
    cos_circular_var = np.cos(fold*circular_var)
    sin_circular_var = np.sin(fold*circular_var)

    # Compute the correlation coefficients
    # r_lc = _corr(linear_var, cos_circular_var)
    # r_ls = _corr(linear_var, sin_circular_var)
    # r_cs = _corr(cos_circular_var, sin_circular_var)

    # pcl = np.sqrt((r_lc**2 + r_ls**2 - 2*r_lc*r_ls*r_cs) / (1 - r_cs**2))

    # Find the regression coefficients for the linear regression
    b_cos = np.linalg.lstsq(
        np.vstack(
            [cos_circular_var, np.ones(len(cos_circular_var))]
        ).T, linear_var, rcond=None
    )[0][0]
    b_sin = np.linalg.lstsq(
        np.vstack(
            [sin_circular_var, np.ones(len(sin_circular_var))]
        ).T, linear_var, rcond=None
    )[0][0]

    # Calculate the amplitude of the fitted sine
    amp = np.sqrt(b_cos**2 + b_sin**2)
    hex = amp / 2

    # return pcl
    return hex


def create_surrogate_distribution(series, values):
    """
    Create a surrogate distribution by shuffling the indices of the input
    arrays.

    Parameters
    ----------
    series : array-like
        Time series data.
    values : array-like
        Corresponding values for each point in the time series.

    Returns
    -------
    surrogate_time_series : array-like
        Shuffled time series data.
    surrogate_values : array-like
        Corresponding values for each point in the shuffled time series.

    Raises
    ------
    ValueError
        If the input arrays have different lengths.

    """
    # Checking if both arrays have the same length
    if len(series) != len(values):
        raise ValueError("Both input arrays should have the same length.")

    # Shuffling the indices of the arrays
    shuffled_indices = np.arange(len(series))
    np.random.shuffle(shuffled_indices)

    # Creating the surrogate distribution using the shuffled indices
    surrogate_time_series = np.array([series[i] for i in shuffled_indices])
    surrogate_values = np.array([values[i] for i in shuffled_indices])

    return surrogate_time_series, surrogate_values


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
        print("trajectory plot exists, skipping")


def get_plotting_grid(
    trajec,
    offs,
    nx=settings.nx,
    ny=settings.ny,
    grsc=settings.grsc,
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
        [
            (pointsax[0][0] + pointsax[1][0])/2,
            (pointsax[0][1] + pointsax[1][1])/2
        ]
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
    x,
    y,
    z=None,
    cmap=plt.get_cmap('copper'),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/
    master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input
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
    Create list of line segments from x and y coordinates, in the correct
    format for LineCollection: an array of the form numlines x (points per
    line) x 2 (x and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def get_t6upperbnd(tort, dt, m):
    return np.sqrt((1 + 2 / (np.exp(18 * tort**2 * dt) - 1)) / m)


def bin_avg(x, y, signal, xbins, ybins):
    """
    Compute the binned averages for a 2D signal.

    Parameters
    ----------
    x : ndarray
        An array containing the x-coordinates of the points in the signal.
    y : ndarray
        An array containing the y-coordinates of the points in the signal.
    signal : ndarray
        An array containing the signal strengths at each point.
    xbins : int or sequence of scalars or str
        Bin edges along the x-axis.
    ybins : int or sequence of scalars or str
        Bin edges along the y-axis.

    Returns
    -------
    avg : ndarray
        The binned averages in the shape (len(x), len(y)).
    xedges : ndarray
        The bin edges along the x axis.
    yedges : ndarray
        The bin edges along the y axis.

    Notes
    -----
    This function uses `numpy.histogram2d` to bin the signal into boxes
    defined by `xbins` and `ybins`. It then computes the average signal
    within each box, replacing any NaNs (which occur if a box contains no
    points) with zero.
    """
    counts, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
    sum_sig, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], weights=signal)
    np.seterr(invalid="ignore")
    avg = sum_sig / counts
    avg = np.nan_to_num(avg)  # replace NaNs with zero
    return avg, xedges, yedges


def smooth_with_gaussian(avg, sigma):
    """
    Apply a Gaussian filter to an array and scale it to retain the maximum
    value of the original array.

    Parameters
    ----------
    avg : ndarray
        The input array to be smoothed.
    sigma : scalar or sequence of scalars
        The standard deviation for Gaussian kernel. The standard deviations of
        the Gaussian filter are given for each axis as a sequence, or as a
        single number, in which case it is equal for all axes.

    Returns
    -------
    out : ndarray
        The array after applying a Gaussian filter and rescaling to retain
        the maximum value of the original array.

    Notes
    -----
    This function uses scipy.ndimage.gaussian_filter to apply a Gaussian
    filter. Smoothing might cause the maximum value to decrease which is often
    undesired. This function scales with a multiplicative factor obtained by
    dividing the original maximum value by the new maximum value to keep the
    maximum value constant pre and post smoothing respectively.
    """
    smoothed = gaussian_filter(avg, sigma=sigma)
    scale_factor = np.amax(avg) \
        / np.amax(smoothed) if np.amax(smoothed) != 0 else 1
    return smoothed * scale_factor
