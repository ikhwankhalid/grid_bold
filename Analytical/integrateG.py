#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:57:35 2021

@author: naomi
"""
from scipy import special
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import random
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class line_integral_of_G():
    def __init__(
            self,
            clustering,
            conjunctive,
            r_min,
            r_max,
            phi_bar,
            g,
            x_off,
            y_off,
            s,
            kappa_h=None,
            sigma_h=None
    ):

        self.clustering = clustering
        self.conjunctive = conjunctive

        self.r_min = r_min
        self.r_max = r_max
        self.phi_bar = phi_bar
        self.g = g
        self.x_off = x_off
        self.y_off = y_off
        self.s = s

        self.kappa_h = kappa_h
        self.sigma_h = sigma_h

        self.a = 4*np.pi/(np.sqrt(3)*self.s)
        self.bx = self.s*self.x_off
        self.by = self.s*self.y_off
        self.cx = np.array([np.sin(self.g), np.sin(
            np.pi/3+self.g), np.sin(2*np.pi/3+self.g)])
        self.cy = np.array([np.cos(self.g), np.cos(
            np.pi/3+self.g), np.cos(2*np.pi/3+self.g)])

        self.d = np.zeros(3)
        self.e = np.zeros(3)
        for k in range(3):
            self.d[k] = self.a * (self.cx[k] * np.cos(self.phi_bar) +
                                  self.cy[k] * np.sin(self.phi_bar))
            self.e[k] = self.a * (self.cx[k] * self.bx + self.cy[k] * self.by)

    def gen_mu_h(self):
        m = np.random.normal(0., self.sigma_h)
        k = np.random.randint(0, 6)
        return np.pi/3 * k + m

    def head_direction(self, theta, mu_h=None):
        if mu_h == None:
            mu_h = self.gen_mu_h()
        return np.exp(self.kappa_h * np.cos(theta - mu_h)) / (special.i0(self.kappa_h))

    def definite_integral(self, mu_h=None):
        integral = self.primitive_of_G_fixed_phi(
            self.r_max) - self.primitive_of_G_fixed_phi(self.r_min)
        if self.conjunctive == True:
            integral = integral * self.head_direction(self.phi_bar, mu_h)
        return integral

    def primitive_of_G_fixed_phi(self, r):
        return r + self.part_1(0, r) + self.part_1(1, r) + self.part_1(2, r) + self.part_2(0, 1, r) + self.part_2(0, 2, r) + self.part_2(1, 2, r) + self.part_3(r)

    def part_1(self, k, r):
        if abs(self.d[k]) < 1e-14:
            return np.cos(self.e[k]) * r
        else:
            return np.sin(self.d[k] * r - self.e[k]) / self.d[k]

    def part_2(self, k1, k2, r):
        if abs(self.d[k1]) < 1e-14 and abs(self.d[k2]) < 1e-14:
            return np.cos(self.e[k1]) * np.cos(self.e[k2]) * r
        elif self.d[k1] == self.d[k2] and self.d[k1] > 1e-14:
            return (np.sin(2*r*self.d[k1] - self.e[k1] - self.e[k2]) + 2*r*self.d[k1] * np.cos(self.e[k1] - self.e[k2])) / (4*self.d[k1])
        elif self.d[k1] == -self.d[k2] and self.d[k1] > 1e-14:
            return (np.sin(2*r*self.d[k1] - self.e[k1] + self.e[k2]) + 2*r*self.d[k1] * np.cos(self.e[k1] + self.e[k2])) / (4*self.d[k1])
        else:
            return (np.sin(r * (self.d[k1] + self.d[k2]) - (self.e[k1] + self.e[k2])) / (self.d[k1] + self.d[k2]) + np.sin(r * (self.d[k1] - self.d[k2]) - (self.e[k1] - self.e[k2])) / (self.d[k1] - self.d[k2])) / 2

    def part_3(self, r):
        if abs(self.d[0] + self.d[1] + self.d[2]) < 1e-14:
            A = np.cos(self.e[0] + self.e[1] + self.e[2]) * r
        else:
            # print(self.d[0] + self.d[1] + self.d[2])
            A = np.sin(self.d[0] + self.d[1] + self.d[2] * r -
                       (self.e[0] + self.e[1] + self.e[2])) / (self.d[0] + self.d
                                                               [1] + self.d[2])

        if abs(self.d[0] + self.d[1] - self.d[2]) < 1e-14:
            B = np.cos(self.e[0] + self.e[1] - self.e[2]) * r
        else:
            B = np.sin(self.d[0] + self.d[1] - self.d[2] * r -
                       (self.e[0] + self.e[1] - self.e[2])) / (self.d[0] + self.d
                                                               [1] - self.d[2])

        if abs(self.d[0] - self.d[1] + self.d[2]) < 1e-14:
            C = np.cos(self.e[0] - self.e[1] + self.e[2]) * r
        else:
            C = np.sin(self.d[0] - self.d[1] + self.d[2] * r -
                       (self.e[0] - self.e[1] + self.e[2])) / (self.d[0] - self.d
                                                               [1] + self.d[2])

        if abs(self.d[0] - self.d[1] - self.d[2]) < 1e-14:
            D = np.cos(self.e[0] - self.e[1] - self.e[2]) * r
        else:
            D = np.sin(self.d[0] - self.d[1] - self.d[2] * r -
                       (self.e[0] - self.e[1] - self.e[2])) / (self.d[0] - self.d
                                                               [1] - self.d[2])
        return (A + B + C + D) / 4


def convert_to_rhombus(x, y):
    return x+0.5*y, np.sqrt(3)/2*y


def generate_phases(clustering, conjunctive, N, meanoff=None, kappa_cl=None):
    if clustering == False:
        ox, oy = np.meshgrid(np.linspace(
            0, 1, int(np.sqrt(N)),
            endpoint=False),
            np.linspace(
            0, 1, int(np.sqrt(N)),
            endpoint=False))
        ox = ox.reshape(1, -1)[0]
        oy = oy.reshape(1, -1)[0]
        N = int(np.sqrt(N))**2
        oxr, oyr = convert_to_rhombus(ox, oy)
    elif clustering == True:
        oxr, oyr = np.empty(N), np.empty(N)
        for j in range(N):
            x_off_orig_cl, y_off_orig_cl = np.random.vonmises(
                2 * np.pi * (meanoff[0] - 0.5),
                kappa_cl) / 2. / np.pi + 0.5, np.random.vonmises(
                2 * np.pi * (meanoff[1] - 0.5),
                kappa_cl) / 2. / np.pi + 0.5
            oxr[j], oyr[j] = convert_to_rhombus(x_off_orig_cl, y_off_orig_cl)
    return oxr, oyr, N


def star_walk_many_cells(
        N, g, oxr, oyr, s, kappa_cl, kappa_h, sigma_h, n_phi, phi_bar, r_min,
        r_max, conj_perc=None):
    mu_h = None
    result_sw = np.zeros(n_phi)

    for i in range(N):
        for ct in range(n_phi):
            integrate = line_integral_of_G(
                clustering, conjunctive, r_min, r_max, phi_bar[ct],
                g, oxr[i],
                oyr[i],
                s, kappa_h, sigma_h)
            if ct == 0 and conjunctive == True:
                mu_h = integrate.gen_mu_h()
            result_sw[ct] += integrate.definite_integral(mu_h)

    return result_sw/abs(r_max-r_min)


def random_walk_many_cells(
        N, oxr, oyr, kappa_cl, kappa_h, sigma_h, n_phi, phi_bar, n_paths,
        path_length):
    mu_h = None
    result_rw = np.zeros(n_phi)

    path_start = np.random.uniform(0., 10.*s, n_paths)
    path_end = path_start + path_length
    x_off_rhomb, y_off_rhomb = np.empty(n_paths), np.empty(n_paths)
    for j in range(n_paths):
        x_off_orig = random.uniform(0., s)
        y_off_orig = random.uniform(0., s)
        x_off_rhomb[j], y_off_rhomb[j] = convert_to_rhombus(
            x_off_orig, y_off_orig)

    for i in range(N):
        for ct in range(n_phi):
            for j in range(n_paths):
                integrate = line_integral_of_G(
                    clustering, conjunctive, path_start[j],
                    path_end[j],
                    phi_bar[ct],
                    g, oxr[i] + x_off_rhomb[j],
                    oyr[i] + y_off_rhomb[j],
                    s, kappa_h, sigma_h)
                if ct == 0 and conjunctive == True:
                    mu_h = integrate.gen_mu_h()
                result_rw[ct] += integrate.definite_integral(mu_h)

    return result_rw/(n_paths*path_length)


##############################################################################

def plot_results(
        X, Y, G, phi_bar, result_sw, result_rw, N, g, meanoff, s, r_min, r_max,
        n_paths, path_length, n_phi, kappa_cl, kappa_h, sigma_h, path,
        def_string):
    fig_width = 40
    fig_height = 10
    fig = plt.figure(figsize=(fig_width, fig_height))

    num_vert_plots = 1
    num_horiz_plots = 3
    width_ratios = [0.35, 0.35, 0.3]
    height_ratios = [1]
    gs = gridspec.GridSpec(num_vert_plots, num_horiz_plots,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios, hspace=0.2, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 3))
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.xaxis.set_major_formatter(
        ticker.FixedFormatter(
            ["0", "0", r"$\pi/3$", r"$2\pi/3$", r"$\pi$", r"$4\pi/3$",
             r"$5\pi/3$", r"$2\pi$"]))
    ax1.plot(phi_bar, result_sw)
    plt.ylim(0, 5000)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Star-like walk', fontsize=24)
    plt.xlabel('Movement direction $\phi$ (rad)', fontsize=20)
    plt.ylabel('Total firing rate (spikes/s)', fontsize=20)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 3))
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.xaxis.set_major_formatter(
        ticker.FixedFormatter(
            ["0", "0", r"$\pi/3$", r"$2\pi/3$", r"$\pi$", r"$4\pi/3$",
             r"$5\pi/3$", r"$2\pi$"]))
    ax1.plot(phi_bar, result_rw)
    plt.ylim(0, 3.5*N)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Random walk', fontsize=24)
    plt.xlabel('Movement direction $\phi$ (rad)', fontsize=20)
    plt.ylabel('Total firing rate (spikes/s)', fontsize=20)

    ax_params = fig.add_subplot(gs[0, 2])
    params_str = '\n'.join((
        r'Parameters:',
        r'$N =$ %d' % N,
        r'$\gamma =$ %.2f' % g,
        r'$x_{off} =$ %.2f' % meanoff[0],
        r'$y_{off} =$ %.2f' % meanoff[1],
        r'$s =$ %.2f' % s,
        r'------------------------------------------------------',
        r'star-like walk:',
        r' integrated for $r$ from %.2f to %.2f' % (r_min, r_max),
        r'------------------------------------------------------',
        r'random walk:',
        r' averaged over %d paths of length %.4f' % (n_paths, path_length),
        r'------------------------------------------------------',
        r'$\phi$ discretized for %d values' % n_phi,
        r'------------------------------------------------------',
    ))
    params_str_clustering = '\n'.join((
        r'',
        r'Structure-function mapping hypothesis:',
        r'$\kappa_{cl} = $ %.2f' % kappa_cl,
    ))
    params_str_conjunctive = '\n'.join((
        r'',
        r'Conjunctive hypothesis',
        r'$\kappa_h = $ %.2f' % kappa_h,
        r'$\sigma_h = $ %.2f' % sigma_h,
    ))
    ax_params.spines['left'].set_visible(False)
    ax_params.set_yticks([])
    ax_params.set_xticks([])
    ax_params.spines['top'].set_visible(False)
    ax_params.spines['right'].set_visible(False)
    ax_params.spines['bottom'].set_visible(False)
    if conjunctive == True:
        total_str = params_str + params_str_conjunctive
    if clustering == True:
        total_str = params_str + params_str_clustering
    else:
        total_str = params_str
    ax_params.text(0., 1.1, total_str,  fontsize=20,
                   verticalalignment='top', linespacing=1.5)

    fig.savefig(path + def_string + '.png')
    return 0


def do_func(
        N, g, meanoff, s, kappa_cl, kappa_h, sigma_h, n_phi, path_length,
        n_paths, r_min, r_max, path, def_string):
    X, Y, G = None, None, None

    phi_bar = np.linspace(0, 2*np.pi, n_phi)

    oxr, oyr, N = generate_phases(
        clustering, conjunctive, N, meanoff, kappa_cl)

    result_sw = star_walk_many_cells(
        N, g, oxr, oyr, s, kappa_cl, kappa_h, sigma_h, n_phi, phi_bar, r_min, r_max)
    result_rw = random_walk_many_cells(
        N, oxr, oyr, kappa_cl, kappa_h, sigma_h, n_phi, phi_bar, n_paths,
        path_length)

    plot_results(
        X, Y, G, phi_bar, result_sw, result_rw, N, g, meanoff, s, r_min, r_max,
        n_paths, path_length, n_phi, kappa_cl, kappa_h, sigma_h, path,
        def_string)

    return result_sw, result_rw


########## Parameters ##########
g = 0
s = 0.3
conjunctive = True
clustering = False
# N = int(1e4)
kappa_h = 4
# kappa_h = 2.5
sigma_h = 3
# sigma_h = 1.5/180*np.pi


r_min = 0
# r_max = 10

pl = 1.

kappa_cl = 0.1
meanoff = (0., 0.)
#################################
n_phi = 360  # only use multiples of 360 or 60
#################################

for N in np.array([1024]):
    path_length = 0.001
    n_paths = 500
    r_max = 10*s
    string_name = 'N{}_nphi{}_pathlength{}_npaths{}_rmin{}_rmax{}'
    def_string = string_name.format(
        N, n_phi, str(path_length).replace('.', '_'),
        n_paths, str(r_min).replace('.', '_'),
        str(r_max).replace('.', '_'))
    if conjunctive:
        path = 'results/conjunctive/'
        if not os.path.exists(path):
            os.makedirs(path)
    elif clustering:
        path = 'results/clustering/'
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path = 'results/none/'
        if not os.path.exists(path):
            os.makedirs(path)

    A_sw, A_rw = do_func(
        N,
        g,
        meanoff,
        s,
        kappa_cl,
        kappa_h,
        sigma_h,
        n_phi,
        path_length,
        n_paths,
        r_min,
        r_max,
        path,
        def_string
    )
