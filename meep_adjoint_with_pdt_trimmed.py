#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:18:16 2022

@author: skenning
"""

from pdt.core import Simulation, Util, ParameterChangelog, Result
from pdt.opt import DesignRegion, MaterialFunction, LegendreTaperMaterialFunction, MinStepOptimizer
from pdt.tools import Render

import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa

import numpy as np

import matplotlib.pyplot as plt

import copy

import scipy

import h5py as h5

Si_index = 3.4
SiO2_index = 1.44

center_wavelength=1.55
gaussian_width=50
wavelengths=np.asarray([1.55])
taper_order=1
taper_w1=.4
taper_w2=1
taper_length=2
straight_length=5
pml_x_thickness=1
pml_y_thickness=1
to_pml_x=1
to_pml_y=1
resolution=64
min_run_time=200

 
# Taper parameters
taper_order = taper_order
taper_w1 = taper_w1
taper_w2 = taper_w2
taper_length = taper_length

# Other simulation parameters
center_wavelength = center_wavelength
gaussian_width = gaussian_width
wavelengths = wavelengths 

straight_length = straight_length
pml_x_thickness = pml_x_thickness
pml_y_thickness = pml_y_thickness
to_pml_x = to_pml_x
to_pml_y = to_pml_y
resolution = resolution
min_run_time = min_run_time

# Simulation setup
# Simulation cell sizing
sx = 2 * (pml_x_thickness + to_pml_x) + taper_length + 2 * straight_length
sy = 2 * (to_pml_y + pml_y_thickness) + taper_w2
cell = mp.Vector3(sx, sy)

# Boundary conditions
boundary_layers = [mp.PML(pml_x_thickness, direction=mp.X),
                   mp.PML(pml_y_thickness, direction=mp.Y)]

# Our device is symmetric, mirror symmetry to speed up computation
symmetries = []#[mp.Mirror(mp.Y)]

# Materials
Si = mp.Medium(index=Si_index)
SiO2 = mp.Medium(index=SiO2_index)

# Geometry
actual_L_straight = (sx - taper_length) / 2

small_wg = mp.Block(mp.Vector3(actual_L_straight, taper_w1), center=mp.Vector3(-taper_length/2 - actual_L_straight/2, 0), material=Si)
large_wg = mp.Block(mp.Vector3(actual_L_straight, taper_w2), center=mp.Vector3(taper_length/2 + actual_L_straight/2, 0), material=Si)

# Sources and such
parity = mp.ODD_Y + mp.EVEN_Z

input_monitor_pt = mp.Vector3(-0.5 * (taper_length + straight_length), 0)
source_pt = mp.Vector3(-0.5 * taper_length - 0.75 * straight_length, 0)
output_monitor_pt = mp.Vector3(0.5 * (taper_length + straight_length), 0)

sources = [mp.EigenModeSource(src=mp.GaussianSource(1/center_wavelength, width=gaussian_width),
                              center=source_pt,
                              size=mp.Vector3(y=(sy-2*pml_y_thickness)),
                              eig_match_freq=True,
                              eig_parity=parity)]

# Design region setup (using the pdt tools)
meep_dr_nx = int(resolution * taper_length)
meep_dr_ny = int(resolution * taper_w2)

taper = LegendreTaperMaterialFunction(taper_order, [taper_length, taper_w2], taper_w1, taper_w2)
design_region = DesignRegion([taper_length, taper_w2], [meep_dr_nx, meep_dr_ny])

# Design region setup (specific to MEEP)
meep_design_variables = mp.MaterialGrid(mp.Vector3(meep_dr_nx,meep_dr_ny),SiO2,Si,grid_type='U_DEFAULT')
meep_design_region = mpa.DesignRegion(meep_design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(taper_length, taper_w2, 0)))

dr_geometry=mp.Block(size=meep_design_region.size, material=meep_design_variables)

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell,
                    boundary_layers=boundary_layers,
                    geometry=[small_wg, large_wg, dr_geometry],
                    sources=sources,
                    eps_averaging=False, # See what this does
                    default_material=SiO2,
                    symmetries=symmetries)

TE0_input = mpa.EigenmodeCoefficient(sim, mp.Volume(center=input_monitor_pt, size=mp.Vector3(y=(sy-2*pml_y_thickness))), mode=1, eig_parity=parity, forward=True)
TE0_output = mpa.EigenmodeCoefficient(sim, mp.Volume(center=output_monitor_pt, size=mp.Vector3(y=(sy-2*pml_y_thickness))), mode=1, eig_parity=parity, forward=True)
ob_list=[TE0_input, TE0_output]

def objective_function(TE0_input_coeff, TE0_output_coeff):  
    #print("objective_function", TE0_input_coeff.shape, TE0_output_coeff.shape)
    #print("objective_function", np.abs(TE0_input_coeff), np.abs(TE0_output_coeff))
    # We want to minimize the radiated power, while maximizing the transmitted power
    radiated = (npa.abs(TE0_input_coeff)**2) - (npa.abs(TE0_output_coeff)**2) / npa.abs(TE0_input_coeff)**2
    transmitted = npa.abs(TE0_output_coeff)**2 / npa.abs(TE0_input_coeff)**2
                    
    # maximize transmitted - radiated
    return npa.mean(transmitted) # Maximize the power in the fundamental mode at the output

# MEEP adjoint setup
opt = mpa.OptimizationProblem(simulation=sim, 
                                   objective_functions=objective_function,
                                   objective_arguments=ob_list,
                                   design_regions=[meep_design_region],
                                   frequencies=1.0/np.asarray(wavelengths),
                                   decay_by=1e-9,
                                   minimum_run_time=min_run_time)

polynomial_coeffs = (0,)
sigma = 0

# Now we update the design region (meep's) and give it a nice plot
design = design_region.evalMaterialFunction(taper, MaterialFunction.arrayToParams('b', polynomial_coeffs), sigma)
opt.update_design([design.transpose().flatten()])

plt.figure()
opt.plot2D(True)
plt.savefig("lts_working_dir/progress_device.png")

# Run the forward and adjoint run
f0, dJ_du = (opt)()
if len(dJ_du.shape) > 1:
    dJ_du = npa.sum(dJ_du, axis=1)
dJ_du = dJ_du.reshape(design_region.N).transpose()

# Compute the gradient with respect to the taper parameters
order, du_db, min_db, all_du_db = design_region.evalMaterialFunctionDerivative(taper, MaterialFunction.arrayToParams('b', polynomial_coeffs), sigma) # Material function vs taper coefficients
dJ_db = np.asarray([np.sum(dJ_du * du_db[i]) for i in range(len(order))])
            

plt.figure()
plt.title("{polynomial_coeffs}".format(polynomial_coeffs=polynomial_coeffs))
plt.imshow(dJ_du)
plt.colorbar()

plt.savefig("lts_working_dir/progress_adjoint.png")
        
print(dJ_db)

dJ_du_corrected = copy.deepcopy(dJ_du)
dJ_du_corrected[np.abs(dJ_du_corrected) > 1] = 0

plt.figure()
plt.title("{polynomial_coeffs}".format(polynomial_coeffs=polynomial_coeffs))
plt.imshow(dJ_du_corrected)
plt.colorbar()