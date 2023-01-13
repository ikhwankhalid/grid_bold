from utils.utils import visualise_traj
from functions.gridfcts import traj, pwliner
import numpy as np
import pickle
import os
import utils.settings as settings
import time
from datetime import timedelta


def sample_traj_rw(
    bounds: list,
    num_trajecs: int = settings.num_trajecs,
    dt: float = settings.dt,
    tmax: float = settings.tmax,
    speed: float = settings.speed,
    dphi: float = settings.dphi,
    init_dir: float = settings.init_dir
):
    idx = 1
    dir = os.path.join(
        settings.rw_loc,
        f"{idx}"
    )

    while idx != num_trajecs + 1:
        if idx == 2:
                start_time = time.monotonic()
        for i in np.arange(len(bounds)):
            f_suffix = f"{bounds[i]}"
            f_suffix = f_suffix.split(".")[0]

            # first do square boundaries
            fname = os.path.join(
                dir,
                f"square_{f_suffix}.pkl"
            )
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if not os.path.exists(fname):
                trajec = traj(
                    dt=dt,
                    tmax=tmax,
                    sp=speed,
                    init_dir=init_dir,
                    dphi=dphi,
                    bound=bounds[i],
                    sq_bound=True
                )
                # plot first 5 sets for visual inspection
                if idx in [1, 2, 3, 4, 5]:
                    visualise_traj(
                        trajec,
                        fname=os.path.join(
                            settings.loc,
                            "plots",
                            "trajectories",
                            "rw",
                            f"raw",
                            f"{idx}",
                            f"square{f_suffix}"
                        )
                    )
                with open(fname, "wb") as f:
                    pickle.dump(trajec, f)
            else:
                print("Trajectory file exists, skipping trajectory sampling.")

            # then do circle boundaries
            fname = os.path.join(
                dir,
                f"circle_{f_suffix}.pkl"
            )
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if not os.path.exists(fname):
                trajec = traj(
                    dt=dt,
                    tmax=tmax,
                    sp=speed,
                    init_dir=init_dir,
                    dphi=dphi,
                    bound=bounds[i],
                    sq_bound=False
                )
                # plot first 5 sets for visual inspection
                if idx in [1, 2, 3, 4, 5]:
                    visualise_traj(
                        trajec,
                        fname=os.path.join(
                            settings.loc,
                            "plots",
                            "trajectories",
                            "rw",
                            f"raw",
                            f"{idx}",
                            f"circle{f_suffix}"
                        )
                    )
                with open(fname, "wb") as f:
                    pickle.dump(trajec, f)
            else:
                print("Trajectory file exists, skipping trajectory sampling.")
        if idx == 2:
                end_time = time.monotonic()
                print(
                    "One simulation finished in: ", 
                    timedelta(seconds=end_time - start_time)
                )
                print(
                    f"{settings.num_trajecs} simulations estimated in: ", 
                    timedelta(
                        seconds=(end_time - start_time) * settings.num_trajecs
                    )
                )
        idx += 1
        dir = os.path.join(
            settings.rw_loc,
            f"{idx}"
        )
    print("Done sampling trajectories.")


def sample_traj_pwlin(
    bounds: list,
    num_trajecs: int = settings.num_trajecs,
    dt: float = settings.dt_pwlin1,
    tmax: float = settings.tmax,
    speed: float = settings.speed,
    dphi: float = settings.dphi,
    init_dir: float = settings.init_dir
):
    idx = 1
    dir = os.path.join(
        settings.pwl_loc,
        f"{idx}"
    )

    while idx != num_trajecs + 1:
        if idx == 2:
                start_time = time.monotonic()
        for i in np.arange(len(bounds)):
            f_suffix = f"{bounds[i]}"
            f_suffix = f_suffix.split(".")[0]

            # first do square boundaries
            fname = os.path.join(
                dir,
                f"square_{f_suffix}.pkl"
            )
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if not os.path.exists(fname):
                t, x, y, direc = traj(
                    dt=dt,
                    tmax=tmax,
                    sp=speed,
                    init_dir=init_dir,
                    dphi=dphi,
                    bound=bounds[i],
                    sq_bound=True
                )
                trajec = pwliner(settings.dt_pwlin2, t, x, y, direc)
                # plot first 5 sets for visual inspection
                if idx in [1, 2, 3, 4, 5]:
                    visualise_traj(
                        trajec,
                        fname=os.path.join(
                            settings.loc,
                            "plots",
                            "trajectories",
                            "pwl",
                            f"raw",
                            f"{idx}",
                            f"square{f_suffix}"
                        )
                    )
                with open(fname, "wb") as f:
                    pickle.dump(trajec, f)
            else:
                print("Trajectory file exists, skipping trajectory sampling.")

            # then do circle boundaries
            fname = os.path.join(
                dir,
                f"circle_{f_suffix}.pkl"
            )
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            if not os.path.exists(fname):
                t, x, y, direc = traj(
                    dt=dt,
                    tmax=tmax,
                    sp=speed,
                    init_dir=init_dir,
                    dphi=dphi,
                    bound=bounds[i],
                    sq_bound=False
                )
                trajec = pwliner(settings.dt_pwlin2, t, x, y, direc)
                # plot first 5 sets for visual inspection
                if idx in [1, 2, 3, 4, 5]:
                    visualise_traj(
                        trajec,
                        fname=os.path.join(
                            settings.loc,
                            "plots",
                            "trajectories",
                            "pwl",
                            f"raw",
                            f"{idx}",
                            f"circle{f_suffix}"
                        )
                    )
                with open(fname, "wb") as f:
                    pickle.dump(trajec, f)
            else:
                print("Trajectory file exists, skipping trajectory sampling.")
        if idx == 2:
                end_time = time.monotonic()
                print(
                    "One simulation finished in: ", 
                    timedelta(seconds=end_time - start_time)
                )
                print(
                    f"{settings.num_trajecs} simulations estimated in: ", 
                    timedelta(
                        seconds=(end_time - start_time) * settings.num_trajecs
                    )
                )
        idx += 1
        dir = os.path.join(
            settings.pwl_loc,
            f"{idx}"
        )
    print("Done sampling trajectories.")


if __name__ == "__main__":
    sample_traj_rw(bounds=settings.bounds)
    sample_traj_pwlin(bounds=settings.bounds)
