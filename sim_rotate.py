"""
Performs rotation simulations for all hypotheses.
"""
from utils.grid_funcs import (
    gridpop_clustering,
    gridpop_repsupp,
    gridpop_conj,
    rotate_traj,
    gen_offsets,
    traj
)
from utils.utils import get_hexsym, get_pathsym, convert_to_rhombus
import numpy as np
import pickle
import os
import settings
import time
from datetime import timedelta
from joblib import Parallel, delayed
from tqdm import tqdm


def compare_rotate_samegrid(
    orient_type: str,
    meanoff_type: str,
    n_trajecs: int = settings.ntrials_finite,
    n_ang: int = settings.n_ang_rotate,
    ang_range: list = settings.ang_range_rotate,
    grsc: float = settings.grsc,
    N: int = settings.N,
    phbins: int = settings.phbins,
    hypothesis: str = None,
    size: float = 60.,
    overwrite: bool = False
):
    # check for valid hypothesis argument
    assert hypothesis in ["repsupp", "clustering", "conjunctive"], \
        "hypothesis must be 'repsupp', 'clustering', or 'conjunctive'"
    # check for valid mean offset argument
    assert meanoff_type in ["uniform", "zero"], \
        "meanoff_type must be 'uniform', or 'zero'"
    assert orient_type in ["random", "zero", "shuffle"], \
        "orient_type must be 'random', 'zero', or 'shuffle"
    print("performing rotation simulations for ", hypothesis)

    circfr_arr = np.zeros((n_trajecs, n_ang, phbins))
    sqfr_arr = np.zeros((n_trajecs, n_ang, phbins))
    circhexes = np.zeros((n_trajecs, n_ang))
    sqhexes = np.zeros((n_trajecs, n_ang))
    circpathhexes = np.zeros((n_trajecs, n_ang))
    sqpathhexes = np.zeros((n_trajecs, n_ang))

    # create list of angles from desired range
    ang_list = np.linspace(ang_range[0], ang_range[1], num=n_ang)

    # orientation of grid axes
    if orient_type == "zero":
        orient = 0.
    else:
        orient = 2*np.pi*np.random.rand()

    shufforient = True if orient_type == "shuffle" else False

    # prepare firing rate and hexasymmetry filenames for saving
    os.makedirs(
        os.path.join(settings.loc, hypothesis, "rotate", meanoff_type),
        exist_ok=True
    )
    circfr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "rotate",
        meanoff_type,
        f"{orient_type}",
        f"circfr_rotate_{size}.pkl"
    )
    sqfr_fname = os.path.join(
        settings.loc,
        hypothesis,
        "rotate",
        meanoff_type,
        f"{orient_type}",
        f"sqfr_rotate_{size}.pkl"
    )
    circhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "rotate",
        meanoff_type,
        f"{orient_type}",
        f"circhex_rotate_{size}.pkl"
    )
    sqhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "rotate",
        meanoff_type,
        f"{orient_type}",
        f"sqhex_rotate_{size}.pkl"
    )
    circpathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "rotate",
        meanoff_type,
        f"{orient_type}",
        f"circpathhex_rotate_{size}.pkl"
    )
    sqpathhex_fname = os.path.join(
        settings.loc,
        hypothesis,
        "rotate",
        meanoff_type,
        f"{orient_type}",
        f"sqpathhex_rotate_{size}.pkl"
    )

    def process_trajec(i):
        trial_data = {
            'circfr_trial': np.zeros((n_ang, phbins)),
            'sqfr_trial': np.zeros((n_ang, phbins)),
            'circhexes_trial': np.zeros((n_ang)),
            'sqhexes_trial': np.zeros((n_ang)),
            'circpathhexes_trial': np.zeros((n_ang)),
            'sqpathhexes_trial': np.zeros((n_ang)),
        }

        if meanoff_type == "zero":
            meanoff = (0, 0)
        elif meanoff_type == "uniform":
            meanoff = (np.random.uniform(0, 1), np.random.uniform(0, 1))
        if hypothesis == "repsupp" or hypothesis == "conjunctive":
            ox, oy = gen_offsets(N=settings.N, kappacl=0., meanoff=meanoff)
        elif hypothesis == "clustering":
            ox, oy = gen_offsets(
                N=settings.N, kappacl=settings.kappa_si, meanoff=meanoff
            )
        oxr, oyr = convert_to_rhombus(ox, oy)
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
            for j, ang in enumerate(ang_list):
                if hypothesis == "clustering":
                    _, mean_fr, _, summed_fr = gridpop_clustering(
                        N,
                        grsc,
                        phbins,
                        traj=rotate_traj(trajec=trajec, theta=ang),
                        oxr=oxr,
                        oyr=oyr
                    )
                elif hypothesis == "conjunctive":
                    _, mean_fr, _, summed_fr = gridpop_conj(
                        N,
                        grsc,
                        phbins,
                        traj=rotate_traj(trajec=trajec, theta=ang),
                        oxr=oxr,
                        oyr=oyr,
                        ang=orient,
                        propconj=settings.propconj_i,
                        kappa=settings.kappac_i,
                        jitter=settings.jitterc_i,
                        shufforient=shufforient
                    )
                else:
                    _, mean_fr, _, summed_fr = gridpop_repsupp(
                        N,
                        grsc,
                        phbins,
                        traj=rotate_traj(trajec=trajec, theta=ang),
                        oxr=oxr,
                        oyr=oyr,
                        tau_rep=settings.tau_rep,
                        w_rep=settings.w_rep
                    )
                if k == 0:
                    trial_data['circfr_trial'][j, :] = mean_fr
                    trial_data['circhexes_trial'][j] = get_hexsym(
                        summed_fr,
                        rotate_traj(trajec=trajec, theta=ang)
                    )
                    trial_data['circpathhexes_trial'][j] = get_pathsym(
                        rotate_traj(trajec=trajec, theta=ang)
                    )
                elif k == 1:
                    trial_data['sqfr_trial'][j, :] = mean_fr
                    trial_data['sqhexes_trial'][j] = get_hexsym(
                        summed_fr,
                        rotate_traj(trajec=trajec, theta=ang)
                    )
                    trial_data['sqpathhexes_trial'][j] = get_pathsym(
                        rotate_traj(trajec=trajec, theta=ang)
                    )

        return trial_data

    # if directory exists, assume data already exists and load
    if not os.path.exists(circfr_fname) or overwrite:
        results = Parallel(n_jobs=50)(
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
        print("rotate simulations already exist")


if __name__ == "__main__":
    start_time = time.monotonic()
    for orient_type in ["random", "shuffle"]:
        # for hypothesis in ["conjunctive", "clustering", "repsupp"]:
        for hypothesis in ["conjunctive"]:
            # for meanoff_type in ["uniform", "zero"]:
            for meanoff_type in ["zero"]:
                for size in settings.rot_sizes:
                    compare_rotate_samegrid(
                        orient_type=orient_type,
                        meanoff_type=meanoff_type,
                        n_trajecs=settings.ntrials_finite,
                        n_ang=settings.n_ang_rotate,
                        ang_range=settings.ang_range_rotate,
                        N=settings.N,
                        hypothesis=hypothesis,
                        size=size
                    )
    end_time = time.monotonic()
    print(
        "Rotation simulations finished in: ",
        timedelta(seconds=end_time - start_time)
    )
