"""
This script calculates and saves the path symmetries of the three trajectory
types: star-like runs, piece-wise linear walks, and random walks. This data
is then loaded whenever the value of a patth symmetry is needed. This way,
path symmetries are only calculated once, which is justified since it is
independent of the hypothesis underlying the formation of grid-like
representations in space.
"""
import numpy as np
from functions.gridfcts import traj, traj_pwl
from utils.utils import get_pathsym
import utils.settings as settings
import os
import pickle


if __name__ == "__main__":
    n = 100
    pathsyms = {}


    # star-like runs
    # no randomness, so only do 1 trajectory
    r,phi,= np.meshgrid(
        np.linspace(0,settings.rmax,settings.bins), 
        np.linspace(0,2 * np.pi,settings.phbins, endpoint=False)
    )
    star_path60 = get_pathsym((1,1,1,phi))
    pathsyms["star"] = star_path60


    # pwl runs
    pwl_path60 = []
    for i in range(n):
        trajec = traj_pwl(
            settings.phbins,
            settings.rmax,
            settings.dt,
            sp=settings.speed
        )
        pwl_path60.append(get_pathsym(trajec))
    pathsyms["pwl"] = pwl_path60


    # random walks
    rw_path60 = []
    for i in range(n):
        trajec = traj(settings.dt, settings.tmax, sp=settings.speed)
        rw_path60.append(get_pathsym(trajec))
    pathsyms["rw"] = rw_path60


    print(
        float('%.3g' % pathsyms["star"]), 
        float('%.3g' % np.mean(pathsyms["pwl"])), 
        float('%.3g' % np.mean(pathsyms["rw"]))
    )


    os.makedirs(os.path.dirname(settings.pathsyms_fname), exist_ok=True)
    with open(settings.pathsyms_fname, 'wb') as f:
        pickle.dump(pathsyms, f)