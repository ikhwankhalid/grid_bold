"""
This script calculates, then prints and plots the attenuation in firing rates
when moving between adjacent grid firing fields under the repetition
suppression hypothesis for different values of the adaptation parameters w_r
and tau_r. This is done for both aligned and misaligned runs. The attenuation
values were used in the manuscript.
"""
import numpy as np
import settings
from utils.utils import convert_to_rhombus, adap_euler
from utils.grid_funcs import grid_meanfr
import matplotlib.pyplot as plt

###############################################################################
# Parameters                                                                  #
###############################################################################
# trajectory parameters
bins = settings.bins * 10
rmax = settings.rmax * 10
part = 4000

# grid cell parameters
N = 1
ox, oy = np.zeros(N), np.zeros(N)
oxr, oyr = convert_to_rhombus(ox, oy)
tt = np.linspace(0, rmax/settings.speed, bins)
ntau = 63
nw = 21
taus = np.round(np.logspace(-1, 2, ntau), 2)
ws = np.round(np.logspace(-1, 0, nw), 2)


###############################################################################
# Functions                                                                   #
###############################################################################
def compute_mean_arrays(tau_rep, w_rep, angle):
    r, phi, indoff = np.meshgrid(
        np.linspace(0, rmax, bins),
        np.array([np.deg2rad(angle)]),
        np.arange(len(ox))
    )
    X, Y = r*np.cos(phi), r*np.sin(phi)
    grids = settings.amax * grid_meanfr(X, Y, offs=(oxr, oyr))
    grids2 = grids.copy()

    for idir in range(N):
        for ic in range(N):
            v = adap_euler(grids[idir, :, ic], tt, tau_rep, w_rep)
            grids[idir, :, ic] = v

    return grids, grids2


def get_attenuation(grids, grids2):
    attenuation = (np.amax(grids2[0, 3333:, 0]) - np.amax(grids[0, 3333:, 0]))\
        / np.amax(grids2[0, 3333:, 0]) * 100

    return attenuation


###############################################################################
# Run                                                                         #
###############################################################################
# initialise vector of attenuations
attenuations = np.zeros((ntau, nw))

# get firing rate with and without attenuation for different parameters
grids_ideal_60, grids2_ideal_60 = compute_mean_arrays(
    settings.tau_rep, settings.w_rep, 60.
)
grids_ideal_30, grids2_ideal_30 = compute_mean_arrays(
    settings.tau_rep, settings.w_rep, 30.
)
grids_real_60, grids2_real_60 = compute_mean_arrays(
    settings.tau_rep / 2, settings.w_rep / 2, 60.
)
grids_real_30, grids2_real_30 = compute_mean_arrays(
    settings.tau_rep / 2, settings.w_rep / 2, 30.
)
grids_halfw_60, grids2_halfw_60 = compute_mean_arrays(
    settings.tau_rep, settings.w_rep / 2, 60.
)
grids_halfw_30, grids2_halfw_30 = compute_mean_arrays(
    settings.tau_rep, settings.w_rep / 2, 30.
)
grids_doubletau_60, grids2_doubletau_60 = compute_mean_arrays(
    settings.tau_rep * 2, settings.w_rep / 2, 60.
)
grids_doubletau_30, grids2_doubletau_30 = compute_mean_arrays(
    settings.tau_rep * 2, settings.w_rep / 2, 30.
)


# print attenuations for all parameters
print("Ideal params:")
print(
    "Attenuation aligned:",
    get_attenuation(grids_ideal_60, grids2_ideal_60)
)
print(
    "Attenuation misaligned:",
    get_attenuation(grids_ideal_30, grids2_ideal_30)
)
print("Real params:")
print(
    "Attenuation aligned:",
    get_attenuation(grids_real_60, grids2_real_60)
)
print(
    "Attenuation misaligned:",
    get_attenuation(grids_real_30, grids2_real_30)
)
print("Half weight same tau params:")
print(
    "Attenuation aligned:",
    get_attenuation(grids_halfw_60, grids2_halfw_60)
)
print(
    "Attenuation misaligned:",
    get_attenuation(grids_halfw_30, grids2_halfw_30)
)
print("Half weight double tau params:")
print(
    "Attenuation aligned:",
    get_attenuation(grids_doubletau_60, grids2_doubletau_60)
)
print(
    "Attenuation misaligned:",
    get_attenuation(grids_doubletau_30, grids2_doubletau_30)
)

###############################################################################
# Plotting                                                                    #
###############################################################################
fig, ax = plt.subplots(4, 1, figsize=(15, 5))
ax[0].plot(grids2_ideal_60[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[0].plot(grids_ideal_60[0, : 666, 0],
           color="black", linestyle="--", label="Ideal")
ax[0].hlines(np.amax(grids_ideal_60[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_ideal_60, grids2_ideal_60)
ax[0].set_title(
    f"tau = {settings.tau_rep}, w = {settings.w_rep}, " +
    f"attenuation = {attn:.0f} %"
)
ax[1].plot(grids2_real_60[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[1].plot(grids_real_60[0, :666, 0], color="black",
           linestyle="--", label="Ideal")
ax[1].hlines(np.amax(grids_real_60[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_real_60, grids2_real_60)
ax[1].set_title(
    f"tau = {settings.tau_rep / 2}, w = {settings.w_rep / 2}, " +
    f"attenuation = {attn:.0f} %"
)
ax[2].plot(grids2_halfw_60[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[2].plot(grids_halfw_60[0, : 666, 0],
           color="black", linestyle="--", label="Ideal")
ax[2].hlines(np.amax(grids_halfw_60[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_halfw_60, grids2_halfw_60)
ax[2].set_title(
    f"tau = {settings.tau_rep}, w = {settings.w_rep / 2}, " +
    f"attenuation = {attn:.0f} %"
)
ax[3].plot(grids2_doubletau_60[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[3].plot(grids_doubletau_60[0, : 666, 0],
           color="black", linestyle="--", label="Ideal")
ax[3].hlines(np.amax(grids_doubletau_60[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_doubletau_60, grids2_doubletau_60)
ax[3].set_title(
    f"tau = {settings.tau_rep * 2}, w = {settings.w_rep / 2}, " +
    f"attenuation = {attn:.0f} %"
)
plt.suptitle("Peak attenuation for aligned runs")
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(4, 1, figsize=(15, 5))
ax[0].plot(grids2_ideal_30[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[0].plot(
    grids_ideal_30[0, :666, 0], color="black", linestyle="--", label="Ideal"
)
ax[0].hlines(np.amax(grids_ideal_30[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_ideal_30, grids2_ideal_30)
ax[0].set_title(
    f"tau = {settings.tau_rep}, w = {settings.w_rep}, " +
    f"attenuation = {attn:.0f} %"
)
ax[1].plot(grids2_real_30[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[1].plot(
    grids_real_30[0, :666, 0], color="black", linestyle="--", label="Ideal"
)
ax[1].hlines(np.amax(grids_real_30[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_real_30, grids2_real_30)
ax[1].set_title(
    f"tau = {settings.tau_rep / 2}, w = {settings.w_rep / 2}, " +
    f"attenuation = {attn:.0f} %"
)
ax[2].plot(grids2_halfw_30[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[2].plot(
    grids_halfw_30[0, :666, 0], color="black", linestyle="--", label="Ideal"
)
ax[2].hlines(np.amax(grids_halfw_30[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_halfw_30, grids2_halfw_30)
ax[2].set_title(
    f"tau = {settings.tau_rep}, w = {settings.w_rep / 2}, " +
    f"attenuation = {attn:.0f} %"
)
ax[3].plot(grids2_doubletau_30[0, :666, 0], color="black",
           linewidth=2., label="No adaptation")
ax[3].plot(
    grids_doubletau_30[0, :666, 0],
    color="black",
    linestyle="--",
    label="Ideal"
)
ax[3].hlines(np.amax(grids_doubletau_30[0, 3333:, 0]), 0, 666)
attn = get_attenuation(grids_doubletau_30, grids2_doubletau_30)
ax[3].set_title(
    f"tau = {settings.tau_rep * 2}, w = {settings.w_rep / 2}, " +
    f"attenuation = {attn:.0f} %"
)
plt.suptitle("Peak attenuation for misaligned runs")
plt.tight_layout()
plt.show()
plt.close()
