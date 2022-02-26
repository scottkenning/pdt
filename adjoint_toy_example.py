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
from autograd import numpy as npa
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

actual_L_straight = (sx - L_taper) / 2

small_wg = mp.Block(mp.Vector3(actual_L_straight, w1), center=mp.Vector3(-L_taper/2 - actual_L_straight/2, 0), material=Si)
large_wg = mp.Block(mp.Vector3(actual_L_straight, w2), center=mp.Vector3(L_taper/2 + actual_L_straight/2, 0), material=Si)

### Simulation setup ###
input_monitor_pt = mp.Vector3(-0.5 * (L_taper + L_straight), 0)
source_pt = mp.Vector3(-0.5 * L_taper - 0.75 * L_straight, 0)
output_monitor_pt = mp.Vector3(0.5 * (L_taper + L_straight), 0)

sources = [mp.EigenModeSource(src=mp.GaussianSource(1/lam, fwidth=0.2/lam),
                              center=source_pt,
                              size=mp.Vector3(y=sy-2*t_pml_y),
                              eig_match_freq=True,
                              eig_parity=mp.ODD_Z+mp.EVEN_Y)]

### Design region setup ###
drr = 100

design_variables = mp.MaterialGrid(mp.Vector3(drr,drr),SiO2,Si,grid_type='U_MEAN')
design_region = mpa.DesignRegion(design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(L_taper, w2, 0)))

dr_geometry=mp.Block(size=design_region.size, material=design_variables)

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell,
                    boundary_layers=boundary_layers,
                    geometry=[small_wg, large_wg, dr_geometry],
                    sources=sources,
                    eps_averaging=False, # See what this does
                    default_material=SiO2)

# Minimize back reflections
TE0_back = mpa.EigenmodeCoefficient(sim, mp.Volume(center=input_monitor_pt, size=mp.Vector3(y=sy-2*t_pml_y)), mode=1, forward=False)
ob_list=[TE0_back]

def J(alpha):
    return npa.abs(alpha)**2

### Setup adjoint problem ###
opt = mpa.OptimizationProblem(simulation=sim, 
                              objective_functions=J,
                              objective_arguments=ob_list,
                              design_regions=[design_region],
                              fcen=lam**-1,
                              df=0,
                              nf=1)

#sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, input_monitor_pt, 1e-9))
x0 = np.random.rand(drr*drr)
opt.update_design([x0])

plt.figure()
opt.plot2D(True)

f0, dJ_du = opt()
plt.figure()
plt.imshow(dJ_du.reshape(drr, drr))


# Looking at the results
# Coefficients are stored sequentially as 2 * nband * nfreqs + nf

#output_res = sim.get_eigenmode_coefficients(output_flux, list(np.arange(1, nbands + 1)), eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha
#input_res = sim.get_eigenmode_coefficients(input_flux, list(np.arange(1, nbands + 1)), eig_parity=mp.ODD_Z + mp.EVEN_Y).alpha



#initial_guess = np.random.uniform(0, 1, k)
#scipy.optimize.minimize(objective, initial_guess, bounds=[(0, 1)]*k, method="L-BFGS-B", options={"ftol" : 0.01, "eps" : 0.01})
#output = scipy.optimize.brute(objective, (slice(0.1, 1, 0.1), slice(0.1, 1, 0.1), slice(0.1, 1, 0.1)))