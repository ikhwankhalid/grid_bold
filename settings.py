"""
Thie file contains all global parameters as well as commonly used filenames
and directories
"""
import os
import numpy as np

###############################################################################
# Plotting parameters                                                         #
###############################################################################
fs = 18                         # fontsize for plots

# reppsupp figure params
imax = 10
# taus = np.round(np.logspace(-1, 2, 6), 2)
# ws = np.round(np.logspace(-1, 0, 6), 2)
taus = np.round(np.logspace(-1, 2, 63), 2)
ws = np.round(np.logspace(-1, 0, 21), 2)

# grid plotting parameters
nx = 4000
ny = 4000
tplot = 300

rep = 60
nstuff = 100
prop_list = np.linspace(1, 100, nstuff, endpoint=True)
smooth_sigma = 0.9

###############################################################################
# Grid cell parameters                                                        #
###############################################################################
num_trajecs = 100
amax = 1.                       # peak / 8 of grid fields
N = int(1024)                   # number of cells to use in simulations
nmax = 200
grsc = 30                       # grid scale in cm
phbins = 360                    # number of bins for phase histogram (N_theta)
phbins_pwl = int(phbins)
idx = 42                        # offset id to highlight
meanoff = (0., 0.)

# conjunctive hypothesis
kappac_i = 50                   # ideal head direction tuning concentration
jitterc_i = 0                   # ideal jitter in head direction tuning
propconj_i = 1                  # ideal proportion of conjunctive cells
kappac_b = 10
jitterc_b = 1.5
kappac_r = 4                    # realistic head direction tuning concentration
jitterc_r = 3                   # realistic jitter in head direction tuning
propconj_r = float(1/3)         # realistic proportion of conjunctive cells

# clustering hypothesis
kappa_si = 10                   # ideal clustering
kappa_sr = 0.1                  # realistic clustering

# repetition suppression
tau_rep = 3.                    # adaptation time constant
w_rep = 1.                      # adaptation strength

###############################################################################
# trajectory parameters                                                       #
###############################################################################
tmax = 9e3                      # max time to simulate in seconds
dphi = 0.5                      # tortuosity parameter
dt = 1e-2                       # time step
speed = 10                      # speed of the agent in cm/s
init_dir = None                 # initial direction of movement
bounds = np.linspace(
    30.,
    180.,
    31
)
bounds = np.hstack(
    (bounds, 1e6)
)
rmax = 300
bins = 1000

# boundary size sims
n_sizes = 51                    # number of environment sizes to sample
size_range = (30., 180.)        # range of environment sizes to sample

# rotation sims
ntrials_finite = 50             # number of trials to average over
n_ang_rotate = 121               # number of rotation angles to sample
ang_range_rotate = [0, 2*np.pi]   # range of rotation angles to sample
rot_sizes = [60.]               # size of finite environment for rotation sims
inf_size = float(1e6)           # size of an "infinite" arena for boundary sims

# tortuosity sims
ntrials_tort = 100              # number of trials to average over
n_torts = 21                    # number of tortuosity values to sample
tort_range = (0.1, 0.7)        # range of tortuosity values to sample

# pwl segment length sims
ntrials_pwl_phbins = 20
max_pwlbins = 14
min_pwlbins = 1
# max_pwlbins = 16
# min_pwlbins = 5
pwl_phbins_range = (min_pwlbins, max_pwlbins)
n_pwl_phbins = max_pwlbins - min_pwlbins + 1

###############################################################################
# file/save paths                                                             #
###############################################################################
# output file path
loc = os.path.realpath(
    os.path.join(
            os.path.dirname(__file__),
            r"data/outputs"
        )
)

conj_offs_fname = os.path.join(
    loc,
    "conjunctive",
    "offsets.pkl"
)
clus_offs_fname = os.path.join(
    loc,
    "clustering",
    "offsets.pkl"
)
reps_offs_fname = os.path.join(
    loc,
    "repsupp",
    "offsets.pkl"
)
pathsyms_fname = os.path.join(
    loc,
    "trajectories",
    "pathsyms.pkl"
)
step_pathsyms_fname = os.path.join(
    loc,
    "trajectories",
    "step_pathsyms.pkl"
)
step_plot_fname = os.path.join(
    loc,
    "trajectories",
    "step_pathsyms.png"
)
step_dict_fname = os.path.join(
    loc,
    "trajectories",
    "step_pathsyms.pkl"
)
m_list_fname = os.path.join(
    loc,
    "trajectories",
    "m_list.pkl"
)

rw_loc = os.path.join(loc, "trajectories", "rw")
pwl_loc = os.path.join(loc, "trajectories", "pwl")
