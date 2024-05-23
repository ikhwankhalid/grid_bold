"""
This script performs the simulations for the data to plot Figure 5 of the
manuscript. Each hypothesis condition is simulated for each trajectory type.
This is done for both "realistic" and "ideal" parameter sets. Each hypothesis
condition is simulated for 300 trials.
"""
import numpy as np
from joblib import Parallel, delayed
import settings
from utils.grid_funcs import (
    traj,
    gen_offsets,
    traj_pwl,
    traj_star,
    gridpop_clustering,
    gridpop_conj,
    gridpop_repsupp
)
from utils.utils import (
    convert_to_rhombus,
    get_hexsym,
    get_pathsym
)
import os
import re
from tqdm import tqdm


###############################################################################
# Parameters                                                                  #
###############################################################################
# hyperparameters
rep = settings.rep
maxrep = 300

# cellular
Ncells = settings.N

####################
# Realistic params #
####################
realistic_params = {
    'propconj': settings.propconj_r,    # fraction of conjunctive cells
    'kappa': settings.kappac_r,         # conj tuning width parameter
    'jitter': settings.jitterc_r,       # conj tuning jitter parameter
    'kappacl': settings.kappa_sr,       # clustering parameter
    'meanoff': settings.meanoff,        # clustering mean offset
    'tau_rep': settings.tau_rep / 2,    # adaptation time constant
    'w_rep': settings.w_rep / 2         # adaptation weight
}

####################
# Ideal parameters #
####################
ideal_params = {
    'propconj': settings.propconj_i,    # fraction of conjunctive cells
    'kappa': settings.kappac_i,         # conj tuning width parameter
    'jitter': settings.jitterc_i,       # conj tuning jitter parameter
    'kappacl': settings.kappa_si,       # clustering parameter
    'meanoff': settings.meanoff,        # clustering mean offset
    'tau_rep': settings.tau_rep,        # adaptation time constant
    'w_rep': settings.w_rep             # adaptation weight
}


###############################################################################
# Functions                                                                   #
###############################################################################
def mfunc(i, params):
    """
    Simulates one trial for each hypothesis condition, for each trajectory
    type. Returns the hexasymmetries, path hexasymmetries, and mean firing
    rates.
    """
    ################
    # Offsets      #
    ################
    # clustering offsets
    ox, oy = gen_offsets(N=settings.N, kappacl=params['kappacl'])
    oxr, oyr = convert_to_rhombus(ox, oy)

    # uniform offsets
    ox2, oy2 = gen_offsets(N=settings.N, kappacl=0.)
    ox2r, oy2r = convert_to_rhombus(ox2, oy2)

    ################
    # random walks #
    ################
    trajec = traj(settings.dt, settings.tmax)

    # conjunctive
    direc_binned, fr_mean, _, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec,
        oxr,
        oyr,
        propconj=params['propconj'],
        kappa=params['kappa'],
        jitter=params['jitter']
    )
    h_conj_rw = (
        get_hexsym(summed_fr, trajec),
        get_pathsym(trajec),
        np.mean(fr_mean)
    )

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_rw = (
        get_hexsym(summed_fr, trajec),
        get_pathsym(trajec),
        np.mean(fr_mean)
    )

    # repsupp
    direc_binned, fr_mean, _, summed_fr = gridpop_repsupp(
        settings.N,
        settings.grsc,
        settings.phbins,
        traj=trajec,
        oxr=ox2r,
        oyr=oy2r,
        tau_rep=params['tau_rep'],
        w_rep=params['w_rep']
    )
    h_rep_rw = (
        get_hexsym(summed_fr, trajec),
        get_pathsym(trajec),
        np.mean(fr_mean)
    )

    ###################
    # star-like walks #
    ###################
    star_offs = np.random.rand(2)
    trajec_star = traj_star(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed,
        offset=star_offs
    )

    # repsupp
    direc_binned, meanfr, _, summed_fr = gridpop_repsupp(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=ox2r,
        oyr=oy2r,
        tau_rep=params['tau_rep'],
        w_rep=params['w_rep']
    )
    h_rep_star = (
        get_hexsym(summed_fr, trajec_star),
        get_pathsym(trajec_star),
        np.mean(meanfr[~np.isnan(meanfr).any()])
    )

    # conj
    direc_binned, meanfr, _, summed_fr = gridpop_conj(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=ox2r,
        oyr=oy2r,
        propconj=params['propconj'],
        kappa=params['kappa'],
        jitter=params['jitter']
    )
    h_conj_star = (
        get_hexsym(summed_fr, trajec_star),
        get_pathsym(trajec_star),
        np.mean(meanfr[~np.isnan(meanfr).any()])
    )

    # clustering
    direc_binned, meanfr, _, summed_fr = gridpop_clustering(
        settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_star,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_star = (
        get_hexsym(summed_fr, trajec_star),
        get_pathsym(trajec_star),
        np.mean(meanfr[~np.isnan(meanfr).any()])
    )

    #####################
    # piece-wise linear #
    #####################
    trajec_pl = traj_pwl(
        settings.phbins,
        settings.rmax,
        settings.dt,
        sp=settings.speed
    )

    # conjunctive
    direc_binned, fr_mean, _, summed_fr = gridpop_conj(
        settings.N,
        settings.grsc,
        settings.phbins,
        trajec_pl,
        ox2r,
        oy2r,
        propconj=params['propconj'],
        kappa=params['kappa'],
        jitter=params['jitter']
    )
    h_conj_pl = (
        get_hexsym(summed_fr, trajec_pl),
        get_pathsym(trajec_pl),
        np.mean(fr_mean[~np.isnan(fr_mean).any()])
    )

    # clustering
    direc_binned, fr_mean, _, summed_fr = gridpop_clustering(
        N=settings.N,
        grsc=settings.grsc,
        phbins=settings.phbins,
        traj=trajec_pl,
        oxr=oxr,
        oyr=oyr
    )
    h_cl_pl = (
        get_hexsym(summed_fr, trajec_pl),
        get_pathsym(trajec_pl),
        np.mean(fr_mean)
    )

    # repsupp
    direc_binned, fr_mean, _, summed_fr = gridpop_repsupp(
        settings.N,
        settings.grsc,
        settings.phbins,
        traj=trajec_pl,
        oxr=ox2r,
        oyr=oy2r,
        tau_rep=params['tau_rep'],
        w_rep=params['w_rep']
    )
    h_rep_pl = (
        get_hexsym(summed_fr, trajec_pl),
        get_pathsym(trajec_pl),
        np.mean(fr_mean)
    )

    return (
        h_conj_star,
        h_conj_rw,
        h_conj_pl,
        h_cl_star,
        h_cl_rw,
        h_cl_pl,
        h_rep_star,
        h_rep_rw,
        h_rep_pl
    )


def get_reps(savename, rep):
    """
    Reads the filesnames in the location and returns a pair of numbers to
    indicate the next sequence of simulatins to be saved.
    """
    # Get present files in the location
    files = os.listdir(os.path.join(settings.loc, "summary", savename))

    # Extract their sequence numbers
    sequence_num = [re.findall(r'_(\d+)-(\d+)', file) for file in files]
    # Flatten the list
    sequence_num = [
        item for sublist in sequence_num for item in sublist
    ]
    sequence_num = [
        tuple(map(int, pair)) for pair in sequence_num if pair
    ]

    # Get the highest sequence number present
    if sequence_num:
        rep_low = max([pair[1] for pair in sequence_num]) + 1
        rep_high = rep_low + rep - 1
    else:
        rep_high = rep
        rep_low = 1

    return rep_low, rep_high


###############################################################################
# Simulate                                                                    #
###############################################################################
savenames = ["real", "ideal"]
for j, params in enumerate([realistic_params, ideal_params]):
    # initialise hex arrays. (gr60, gr60_path, gr0)
    h_cl_star = np.zeros((3, rep))
    h_cl_rw = np.zeros((3, rep))
    h_cl_pl = np.zeros((3, rep))
    h_conj_star = np.zeros((3, rep))
    h_conj_rw = np.zeros((3, rep))
    h_conj_pl = np.zeros((3, rep))
    h_rep_star = np.zeros((3, rep))
    h_rep_rw = np.zeros((3, rep))
    h_rep_pl = np.zeros((3, rep))

    # Define the directory name
    dir_name = os.path.join(settings.loc, "summary", savenames[j])

    # Check if the directory exists
    if not os.path.exists(dir_name):
        # If not, create it
        os.makedirs(dir_name)

    rep_low, rep_high = get_reps(savenames[j], rep)

    while rep_high < maxrep:
        print(f"Doing sims {rep_low} to {rep_high}")
        alldata = Parallel(
            n_jobs=60, timeout=99999, verbose=100
        )(delayed(mfunc)(i, params) for i in tqdm(range(rep)))
        alldata = np.moveaxis(np.moveaxis(np.array(alldata), 1, 0), 1, -1)

        (
            h_conj_star,
            h_conj_rw,
            h_conj_pl,
            h_cl_star,
            h_cl_rw,
            h_cl_pl,
            h_rep_star,
            h_rep_rw,
            h_rep_pl
        ) = alldata

        # same figure for realistic parameter choices
        data = np.array(
            [
                h_conj_star[0, :], h_conj_star[1, :], h_conj_star[2, :],
                h_conj_pl[0, :], h_conj_pl[1, :], h_conj_pl[2, :],
                h_conj_rw[0, :], h_conj_rw[1, :], h_conj_rw[2, :],
                h_cl_star[0, :], h_cl_star[1, :], h_cl_star[2, :],
                h_cl_pl[0, :], h_cl_pl[1, :], h_cl_pl[2, :],
                h_cl_rw[0, :], h_cl_rw[1, :], h_cl_rw[2, :],
                h_rep_star[0, :], h_rep_star[1, :], h_rep_star[2, :],
                h_rep_pl[0, :], h_rep_pl[1, :], h_rep_pl[2, :],
                h_rep_rw[0, :], h_rep_rw[1, :], h_rep_rw[2, :]
            ]
        )

        # Define the full file name
        filename = os.path.join(
            dir_name,
            f"summary_bar_plot_3hyp3traj_{rep_low}-{rep_high}"
        )

        # Save the data
        np.save(filename, data)
        rep_low += rep
        rep_high += rep
