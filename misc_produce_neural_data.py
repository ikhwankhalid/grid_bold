from utils.grid_funcs import (
    gridpop_clustering,
    gridpop_repsupp,
    gridpop_conj,
    gen_offsets,
    traj_star,
    traj_star_delay
)
from utils.utils import get_hexsym, get_pathsym, convert_to_rhombus
import numpy as np
import pickle
import os
import settings

data = {
    "path": {},
    "conj": {},
    "clustering": {},
    "repsupp": {},
}
phbins = 36
delays = ["0", "0", "3"]

ox_uni, oy_uni = gen_offsets(N=settings.N, kappacl=0.)
ox_clust, oy_clust = gen_offsets(
    N=settings.N, kappacl=settings.kappa_si
)
oxr_uni, oyr_uni = convert_to_rhombus(ox_uni, oy_uni)
oxr_clust, oyr_clust = convert_to_rhombus(ox_clust, oy_clust)

if True:
    for i, delay in enumerate(delays):
        if delay == "0":
            if i == 0:
                data_fname = os.path.join(
                    settings.loc,
                    "ext_data",
                    f"data_{int(delay)}_nodel.npy"
                )
                trajec = traj_star(
                        phbins,
                        settings.rmax,
                        settings.dt,
                        sp=settings.speed
                )
            if i == 1:
                data_fname = os.path.join(
                    settings.loc,
                    "ext_data",
                    f"data_{int(delay)}.npy"
                )
                trajec = traj_star_delay(
                        phbins,
                        settings.rmax,
                        settings.dt,
                        sp=settings.speed,
                        delay=float(delay)
                )
        elif delay == "3":
            data_fname = os.path.join(
                settings.loc,
                "ext_data",
                f"data_{int(delay)}.npy"
            )
            trajec = traj_star_delay(
                    phbins,
                    settings.rmax,
                    settings.dt,
                    sp=settings.speed,
                    delay=float(delay)
            )
        data["path"]["t"] = trajec[0]
        data["path"]["x"] = trajec[1]
        data["path"]["y"] = trajec[2]
        data["path"]["direc"] = trajec[3]
        data["path"]["pathhex"] = get_pathsym(trajec)
        print(data["path"]["pathhex"])

        direc_binned, fr_mean, fr, summed_fr = gridpop_conj(
            settings.N,
            settings.grsc,
            settings.phbins,
            traj=trajec,
            oxr=oxr_uni,
            oyr=oyr_uni,
            propconj=settings.propconj_i,
            kappa=settings.kappac_i,
            jitter=settings.jitterc_i
        )
        data["conj"]["summed_fr"] = summed_fr
        data["conj"]["hex"] = get_hexsym(summed_fr, trajec)
        print(data["conj"]["hex"])

        direc_binned, fr_mean, fr, summed_fr = gridpop_repsupp(
            settings.N,
            settings.grsc,
            settings.phbins,
            traj=trajec,
            oxr=oxr_uni,
            oyr=oyr_uni,
            tau_rep=settings.tau_rep,
            w_rep=settings.w_rep
        )
        data["repsupp"]["summed_fr"] = summed_fr
        data["repsupp"]["hex"] = get_hexsym(summed_fr, trajec)
        print(data["repsupp"]["hex"])

        direc_binned, fr_mean, fr, summed_fr = gridpop_clustering(
            settings.N,
            settings.grsc,
            settings.phbins,
            traj=trajec,
            oxr=oxr_clust,
            oyr=oyr_clust
        )
        data["clustering"]["summed_fr"] = summed_fr
        data["clustering"]["hex"] = get_hexsym(summed_fr, trajec)
        print(data["clustering"]["hex"])

        os.makedirs(os.path.dirname(data_fname), exist_ok=True)
        with open(data_fname, "wb") as f:
            pickle.dump(data, f)


for i, delay in enumerate(delays):
    if delay == "0":
        if i == 0:
            data_fname = os.path.join(
                settings.loc,
                "ext_data",
                f"data_{int(delay)}_nodel.npy"
            )
        if i == 1:
            data_fname = os.path.join(
                settings.loc,
                "ext_data",
                f"data_{int(delay)}.npy"
            )
    elif delay == "3":
        data_fname = os.path.join(
            settings.loc,
            "ext_data",
            f"data_{int(delay)}.npy"
        )

    with open(data_fname, "rb") as f:
        data = np.load(data_fname, allow_pickle=True)

    if i > 0:
        delay_steps = int(float(delay) / settings.dt)
        nlin = int(settings.rmax / settings.speed / settings.dt)

        x = data["path"]["x"]
        y = data["path"]["x"]
        for i in range(phbins):
            xdat = x[
                i * (nlin + delay_steps):i * (nlin + delay_steps) + delay_steps
            ]
            assert all(xdat == 0.)
            ydat = y[
                i * (nlin + delay_steps):i * (nlin + delay_steps) + delay_steps
            ]
            assert all(ydat == 0.)
        print("Tests passed")

# data = {
#     # dictionary of path data
#     "path": {
#         "t": [...],            # time
#         "x": [...],            # x coordinates
#         "y": [...],            # y coordinates
#         "direc": [...],        # heading in radians
#         "pathhex": [...]       # path hexasymmetry
#     },
#     # dictionary of conjunctive by head-direction cells hypothesis data
#     "conj": {
#         "summed_fr": [...],    # summed population activity across M time
#                                  steps
#         "hex": [...]           # hexasymmetry
#     },
#     # dictionary of structure-function mapping data
#     "clustering": {
#         "summed_fr": [...],    # summed population activity across M time
#                                  steps
#         "hex": [...]           # hexasymmetry
#     },
#     # dictionary of repetition suppression data
#     "repsupp": {
#         "summed_fr": [...],    # summed population activity across M time
#                                  steps
#         "hex": [...]           # hexasymmetry
#     },
# }
