"""
Performs size simulations for all hypotheses.
"""
from utils.grid_funcs import (
    gridpop_clustering,
    gridpop_conj,
    gridpop_repsupp,
    gen_offsets,
    traj
)
from utils.utils import (
    convert_to_rhombus,
    get_hexsym,
    get_pathsym
)
import numpy as np
import pickle
import os
import settings
import time
from datetime import timedelta
from joblib import Parallel, delayed
from tqdm import tqdm


def sim_bnds(
    traj_type: str,
    meanoff_type: str,
    n_trajecs: int = settings.ntrials_finite,
    n_sizes: tuple = settings.n_sizes,
    size_range: tuple = settings.size_range,
    grsc: float = settings.grsc,
    N: int = settings.N,
    phbins: int = settings.phbins,
    offs_idx: int = 50,
    hypothesis: str = None
):
    # check for valid hypothesis argument
    assert hypothesis in ["repsupp", "clustering", "conjunctive"], \
        "hypothesis must be 'repsupp', 'clustering', or 'conjunctive'"
    print("performing sizes simulations for ", hypothesis)

    # append boundary sizes array with "infinite" boundaries
    sizes = np.append(
        np.linspace(size_range[0], size_range[1], n_sizes),
        settings.inf_size
    )

    # output arrays initialisation
    circfr_arr = np.zeros((n_trajecs, n_sizes + 1, phbins))
    sqfr_arr = np.zeros((n_trajecs, n_sizes + 1, phbins))
    circhexes = np.zeros((n_trajecs, n_sizes + 1))
    sqhexes = np.zeros((n_trajecs, n_sizes + 1))
    circpathhexes = np.zeros((n_trajecs, n_sizes + 1))
    sqpathhexes = np.zeros((n_trajecs, n_sizes + 1))

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(settings.loc, hypothesis, "sizes", meanoff_type),
        exist_ok=True
    )
    circfr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        meanoff_type,
        f"{traj_type}",
        "circfr_sizes.pkl"
    )
    sqfr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        meanoff_type,
        f"{traj_type}",
        "sqfr_sizes.pkl"
    )
    circhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        meanoff_type,
        f"{traj_type}",
        "circhex_sizes.pkl"
    )
    sqhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        meanoff_type,
        f"{traj_type}",
        "sqhex_sizes.pkl"
    )
    circpathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        meanoff_type,
        f"{traj_type}",
        "circpathhex_sizes.pkl"
    )
    sqpathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "sizes",
        meanoff_type,
        f"{traj_type}",
        "sqpathhex_sizes.pkl"
    )

    def process_trajec(i):
        trial_data = {
            'circfr_trial': np.zeros((n_sizes + 1, phbins)),
            'sqfr_trial': np.zeros((n_sizes + 1, phbins)),
            'circhexes_trial': np.zeros((n_sizes + 1)),
            'sqhexes_trial': np.zeros((n_sizes + 1)),
            'circpathhexes_trial': np.zeros((n_sizes + 1)),
            'sqpathhexes_trial': np.zeros((n_sizes + 1)),
        }

        if meanoff_type == "zero":
            meanoff = (0, 0)
        elif meanoff_type == "uniform":
            meanoff = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        if hypothesis == "repsupp" or hypothesis == "conjunctive":
            ox, oy = gen_offsets(N=N, kappacl=0., meanoff=meanoff)
        elif hypothesis == "clustering":
            ox, oy = gen_offsets(
                N=settings.N, kappacl=settings.kappa_si, meanoff=meanoff
            )
        oxr, oyr = convert_to_rhombus(ox, oy)
        for j, size in enumerate(sizes):
            circle_trajec = traj(
                dt=settings.dt,
                tmax=settings.tmax,
                sp=settings.speed,
                init_dir=settings.init_dir,
                dphi=settings.dphi,
                bound=size,
                sq_bound=False
            )
            square_trajec = traj(
                dt=settings.dt,
                tmax=settings.tmax,
                sp=settings.speed,
                init_dir=settings.init_dir,
                dphi=settings.dphi,
                bound=size,
                sq_bound=True
            )
            trajecs = [circle_trajec, square_trajec]
            for k, trajec in enumerate(trajecs):
                if hypothesis == "clustering":
                    _, mean_fr, _, summed_fr = gridpop_clustering(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr
                    )
                elif hypothesis == "conjunctive":
                    _, mean_fr, _, summed_fr = gridpop_conj(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr,
                        propconj=settings.propconj_i,
                        kappa=settings.kappac_i,
                        jitter=settings.jitterc_i
                    )
                else:
                    _, mean_fr, _, summed_fr = gridpop_repsupp(
                        N,
                        grsc,
                        phbins,
                        traj=trajec,
                        oxr=oxr,
                        oyr=oyr,
                        tau_rep=settings.tau_rep,
                        w_rep=settings.w_rep
                    )

                if k == 0:
                    trial_data['circfr_trial'][j, :] = mean_fr
                    trial_data['circhexes_trial'][j] = get_hexsym(
                        summed_fr,
                        trajec
                    )
                    trial_data['circpathhexes_trial'][j] = get_pathsym(
                        trajec
                    )
                elif k == 1:
                    trial_data['sqfr_trial'][j, :] = mean_fr
                    trial_data['sqhexes_trial'][j] = get_hexsym(
                        summed_fr,
                        trajec
                    )
                    trial_data['sqpathhexes_trial'][j] = get_pathsym(
                        trajec
                    )

        return trial_data

    # if firing rate array pickle files don't exist, run simulations
    if not os.path.exists(circfr_fname) and not os.path.exists(sqfr_fname):
        results = Parallel(n_jobs=25)(
            delayed(
                process_trajec
            )(i) for i in tqdm(range(n_trajecs))
        )

        # Combine results
        for i, result in enumerate(results):
            circfr_arr[i, :, :] = result['circfr_trial']
            sqfr_arr[i, :, :] = result['sqfr_trial']
            circhexes[i, :] = result['circhexes_trial']
            sqhexes[i, :] = result['sqhexes_trial']
            circpathhexes[i, :] = result['circpathhexes_trial']
            sqpathhexes[i, :] = result['sqpathhexes_trial']

        os.makedirs(os.path.dirname(circfr_fname), exist_ok=True)
        os.makedirs(os.path.dirname(sqfr_fname), exist_ok=True)
        os.makedirs(os.path.dirname(circhex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(sqhex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(circpathhex_fname), exist_ok=True)
        os.makedirs(os.path.dirname(sqpathhex_fname), exist_ok=True)
        with open(circfr_fname, "wb") as f:
            pickle.dump(circfr_arr, f)
        with open(sqfr_fname, "wb") as f:
            pickle.dump(sqfr_arr, f)
        with open(circhex_fname, "wb") as f:
            pickle.dump(circhexes, f)
        with open(sqhex_fname, "wb") as f:
            pickle.dump(sqhexes, f)
        with open(circpathhex_fname, "wb") as f:
            pickle.dump(circpathhexes, f)
        with open(sqpathhex_fname, "wb") as f:
            pickle.dump(sqpathhexes, f)
    else:
        print("sizes simulations already exist")


if __name__ == "__main__":
    start_time = time.monotonic()
    for traj_type in ["rw"]:
        for hypothesis in ["conjunctive", "clustering", "repsupp"]:
            for meanoff_type in ["uniform"]:
                sim_bnds(
                    traj_type=traj_type,
                    meanoff_type=meanoff_type,
                    n_trajecs=settings.ntrials_finite,
                    hypothesis=hypothesis
                )
    end_time = time.monotonic()
    print(
        "Sizes simulations finished in: ",
        timedelta(seconds=end_time - start_time)
    )
