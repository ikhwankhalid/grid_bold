"""
This script
"""
import numpy as np
from functions.gridfcts import traj, traj_pwl
from utils.utils import get_pathsym
import utils.settings as settings
import os
import pickle
from numba import jit, njit
import matplotlib.pyplot as plt


@jit
def get_step_pathsyms(n, m_list, dphi=settings.dphi):
    # random walks
    pathsyms = np.zeros([n, len(m_list)])
    tmax_list = m_list * settings.dt
    for i in range(n):
        print(i)
        for j, m in enumerate(m_list):
            tmax = settings.dt * m
            trajec = traj(
                settings.dt,
                tmax,
                sp=settings.speed,
                dphi=dphi
            )
            pathsyms[i, j] = get_pathsym(trajec)

    print(len(trajec[-1]))
    print(tmax_list)

    return pathsyms


if __name__ == "__main__":
    n = 500
    # m_list = np.logspace(0, 5, 100, base=10.).astype(int)
    m_list = np.logspace(0, 3, 100, base=10.).astype(int)
    print(m_list)

    tort_list = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
    pathsyms1 = get_step_pathsyms(n, m_list, tort_list[0])
    pathsyms2 = get_step_pathsyms(n, m_list, tort_list[1])
    pathsyms3 = get_step_pathsyms(n, m_list, tort_list[2])
    pathsyms4 = get_step_pathsyms(n, m_list, tort_list[3])
    pathsyms5 = get_step_pathsyms(n, m_list, tort_list[4])
    pathsyms6 = get_step_pathsyms(n, m_list, tort_list[5])
    pathsyms7 = get_step_pathsyms(n, m_list, tort_list[6])
    pathsyms8 = get_step_pathsyms(n, m_list, tort_list[7])
    pathsyms9 = get_step_pathsyms(n, m_list, tort_list[8])
    pathsyms10 = get_step_pathsyms(n, m_list, tort_list[9])
    pathsyms11 = get_step_pathsyms(n, m_list, tort_list[10])

    os.makedirs(os.path.dirname(settings.step_pathsyms_fname), exist_ok=True)
    with open(settings.step_pathsyms_fname, 'wb') as f:
        pickle.dump(pathsyms1, f)

    os.makedirs(os.path.dirname(settings.m_list_fname), exist_ok=True)
    with open(settings.m_list_fname, 'wb') as f:
        pickle.dump(m_list, f)

    x = lambda a : 1 / np.sqrt(a)

    plt.figure(figsize=(24,14))
    plt.rcParams.update({'font.size': settings.fs})
    plt.scatter(
        m_list,
        np.median(pathsyms1, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = 0.$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms2, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .01$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms3, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .02$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms4, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .03$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms5, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .04$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms6, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .05$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms7, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .06$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms8, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .07$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms9, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .08$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms10, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .09$"
    )
    plt.scatter(
        m_list,
        np.mean(pathsyms11, axis=0),
        label=r"$|\overset{\sim}{T}|_6$, $\sigma_\theta = .1$"
    )
    plt.plot(
        m_list,
        x(m_list),
        label=r"$\frac{1}{\sqrt{M}}$",
        color="black",
        linestyle="dotted",
        linewidth=3
    )
    plt.plot(
        m_list, 
        np.ones(len(m_list)),
        label=r"$T_0$",
        color="black",
        linestyle="--",
        linewidth=3
    )
    plt.ylabel("Path hexasymmetry")
    plt.xlabel("Number of time steps M")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([10**i for i in range(0,6)])
    plt.xlim(8 * 10**-1, 2 * 10**5)
    plt.title("Mean over {} random walks".format(n))
    plt.legend()
    plt.tight_layout()
    plt.savefig(settings.step_plot_fname)

tort_pathsyms = {}
pathlist = [pathsyms1, pathsyms2, pathsyms3, pathsyms4, pathsyms5, pathsyms6, pathsyms7, pathsyms8, pathsyms9, pathsyms10, pathsyms11]
for i, path in enumerate(pathlist):
    tort_pathsyms[f"{np.round(0. + i*0.01, 2)}"] = path

tort_pathsyms["m_list"] = m_list

os.makedirs(os.path.dirname(settings.step_dict_fname), exist_ok=True)
with open(settings.step_dict_fname, 'wb') as f:
    pickle.dump(tort_pathsyms, f)