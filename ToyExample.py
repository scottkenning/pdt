#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:33:05 2022

@author: skenning
"""

"""
This example parameterizes a taper and picks coefficients for the width of a
taper. We utilize the odd indexed (zero-indexed) Legendre polynomials 
in a linear-combination fashion to form the basis for the width.
"""

from scipy.special import legendre
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import meep as mp
import meep.adjoint as mpa

# Log used for optimization
import logging
logging.basicConfig(filename='ToyExample.log', encoding='utf-8', level=logging.DEBUG)

### Variables you can play with ###
# We wish to have k degrees of freedom with our functions
k = 3

# Input and output widths
w1 = 1
w2 = 5

# Length of different WG sections
L_taper = 3
L_straight = 6

# PML
t_pml_x = 3
t_pml_y = 2
to_pml = 3

# Frequency
lam = 1.55

### Setup for the width generation function ###
# Zero indexed, so pick the odd legendre polynomials
basis = [legendre(2*n + 1) for n in range(k)] 

# Some useful computations
w_travel = (w2 - w1) / 2

counter = 0
def objective(beta):
    global counter
    ### Setup for meep ###
    resolution = 10
    
    sx = 2 * (t_pml_x + to_pml) + L_taper + 2 * L_straight
    sy = 2 * (to_pml + t_pml_y) + w2
    cell = mp.Vector3(sx, sy)
    
    boundary_layers = [mp.PML(t_pml_x, direction=mp.X),
                       mp.PML(t_pml_y, direction=mp.Y)]
    
    symmetries = [mp.Mirror(mp.Y)]
    
    Si = mp.Medium(index=3.4)
    SiO2 = mp.Medium(index=1.44)
    
    # Evaluation of the basis functions
    def eval_weighted(beta, s):
        val = np.zeros(len(s))
        eval_basis = lambda i: w_travel * (basis[i](s) + 1) / 2

        # Evaluate for the degrees of freedom
        for i in range(k):
            val += beta[i] * eval_basis(i)
            
        # We mutiply everything by a scaling factor to assure that the basis values
        # sum to 1
        val /= np.sum(beta)
            
        return val
    def get_taper(beta, length):
        s = np.linspace(-1, 1, 2 * resolution * max(w_travel, L_taper))
        x = s * length / 2
        y = eval_weighted(beta, s)
        
        return (x, y)

    # Plot out the basis functions
    def plot_basis():
        plt.figure()
        plt.title("unweighted basis")
        x = np.linspace(-1, 1, 1000)
        
        for b in basis:
            plt.plot(x, b(x))
    def plot_weighted(beta):
        plt.figure()
        plt.title("weighted basis")
        x = np.linspace(-1, 1, 1000)
        plt.plot(x, eval_weighted(beta, x))
    def plot_waveguide_bounds(beta, length):
        plt.figure()
        plt.title("waveguide bounds")
        
        s = np.linspace(-1, 1, 1000)
        x = s * length / 2
        y = eval_weighted(beta, s)
        
        plt.plot(x, y + w1 / 2)
        plt.plot(x, -y - w1 / 2)
    
    ### Meep geometry ###
    def generate_geometry(beta):
        vertices = []
        
        # Bottom left straight section
        vertices.append(mp.Vector3(-0.5 * sx, -0.5 * w1))
        
        # Generate the bottom vertices of the structure
        taper_x, taper_y = get_taper(beta, L_taper)
        taper_y = -taper_y - w1 / 2
        for i in range(len(taper_x)):
            vertices.append(mp.Vector3(taper_x[i], taper_y[i]))
        
        # Bottom right straight section
        vertices.append(mp.Vector3(0.5 * sx, -0.5 * w2))
        
        # Top right straight section
        vertices.append(mp.Vector3(0.5 * sx, 0.5 * w2))
        
        # Generate the bottom vertices of the structure
        taper_x, taper_y = get_taper(beta, L_taper)
        taper_y = taper_y + w1 / 2
        taper_x = np.flip(taper_x)
        taper_y = np.flip(taper_y)
        for i in range(len(taper_x)):
            vertices.append(mp.Vector3(taper_x[i], taper_y[i]))
        
        # Top left straight section
        vertices.append(mp.Vector3(-0.5 * sx, 0.5 * w1))
        
        return vertices
    
    vertices = generate_geometry(beta)
    
    ### Simulation setup ###
    input_monitor_pt = mp.Vector3(-0.5 * (L_taper + L_straight), 0)
    source_pt = mp.Vector3(-0.5 * L_taper - 0.75 * L_straight, 0)
    output_monitor_pt = mp.Vector3(0.5 * (L_taper + L_straight), 0)
    
    sources = [mp.EigenModeSource(src=mp.GaussianSource(1/lam, fwidth=0.2/lam),
                                  center=source_pt,
                                  size=mp.Vector3(y=sy-2*t_pml_y),
                                  eig_match_freq=True,
                                  eig_parity=mp.ODD_Z+mp.EVEN_Y)]
    
    sim = mp.Simulation(resolution=resolution,
                        cell_size=cell,
                        boundary_layers=boundary_layers,
                        geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
                        sources=sources,
                        symmetries=symmetries)
    
    # df, nfreqs
    nfreqs = 10
    nbands = 5
    freqs = np.linspace((1/(lam + 0.1)), (1/(lam - 0.1)), nfreqs)
    input_flux = sim.add_flux(1/lam, ((1/(lam - 0.1)) - (1/(lam + 0.1))) / nfreqs, nfreqs, mp.FluxRegion(center=input_monitor_pt, size=mp.Vector3(y=sy-2*t_pml_y)))
    output_flux = sim.add_flux(1/lam, ((1/(lam - 0.1)) - (1/(lam + 0.1))) / nfreqs, nfreqs, mp.FluxRegion(center=output_monitor_pt, size=mp.Vector3(y=sy-2*t_pml_y)))
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, input_monitor_pt, 1e-9))
    sim.plot2D()
    plt.savefig("progress/{counter}.png".format(counter=counter))
    
    # Looking at the results
    # Coefficients are stored sequentially as 2 * nband * nfreqs + nf
    
    output_res = sim.get_eigenmode_coefficients(output_flux, list(np.arange(1, nbands + 1)), eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha
    input_res = sim.get_eigenmode_coefficients(input_flux, list(np.arange(1, nbands + 1)), eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha
    
    #coeffs[band, freq, direction]
    def print_transmission_scatter(res):
        S = pd.DataFrame()
        
        freq_bin = []
        band_bin = []
        dir_bin = []
        
        scatter_bin = []
        
        for i, freq in np.ndenumerate(freqs):
            for j in np.arange(0, nbands):
                for d in range(2):
                    freq_bin.append(freq)
                    band_bin.append(j)
                    dir_bin.append(d)
                    
                    scatter_bin.append(np.abs(res[j, i, d][0])**2)
            
        S["frequency"] = freq_bin
        S["band"] = band_bin
        S["direction"] = dir_bin
        S["scatter power"] = scatter_bin
        
        print(S)
        
    def get_scatter_higher_modes(res, direction):
        metrics = []
        
        for i, freq in np.ndenumerate(freqs):
            for j in np.arange(1 - direction, nbands):
                metrics.append(np.abs(res[j, i, direction][0])**2)
                
        return metrics
       
    print_transmission_scatter(input_res)
    print_transmission_scatter(output_res)
    
    # Figure of merit is the maximum scattered power to higher order modes
    fom = np.max(get_scatter_higher_modes(output_res, 0)) #max(np.max(get_scatter_higher_modes(output_res, 0)), np.max(get_scatter_higher_modes(input_res, 1)))
        
    sim.reset_meep()
    
    counter += 1
    
    logging.info("{counter}: {beta} -> {fom}".format(counter=counter, beta=str(beta), fom=fom))
    
    return fom

#initial_guess = np.random.uniform(0, 1, k)
#scipy.optimize.minimize(objective, initial_guess, bounds=[(0, 1)]*k, method="L-BFGS-B", options={"ftol" : 0.01, "eps" : 0.01})
output = scipy.optimize.brute(objective, (slice(0.1, 1, 0.1), slice(0.1, 1, 0.1), slice(0.1, 1, 0.1)))