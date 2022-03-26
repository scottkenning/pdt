#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:18:16 2022

@author: skenning
"""

from pdt.core import Simulation, Util, ParameterChangelog, Result
from pdt.opt import DesignRegion, MaterialFunction, LegendreTaperMaterialFunction, MinStepOptimizer, ScipyGradientOptimizer
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

class LegendreTaperSimulation(Simulation):
    def __init__(self, 
                 center_wavelength,
                 gaussian_width,
                 wavelengths,
                 taper_order, 
                 taper_w1, 
                 taper_w2, 
                 taper_length,
                 optimize=True,
                 catch_errors=False):
        Simulation.__init__(self, "WaveguideTaperFD", working_dir="WaveguideTaperFD_wd", catch_errors=catch_errors)
        
        # Taper parameters
        self.taper_order = taper_order
        self.taper_w1 = taper_w1
        self.taper_w2 = taper_w2
        self.taper_length = taper_length
        
        # Other simulation parameters
        self.center_wavelength = center_wavelength
        self.gaussian_width = gaussian_width
        self.wavelengths = wavelengths 
        
        # Turn on/off optimization: debugging
        self.optimize = optimize
        
    def run(self, parameters):
        # Convergence parameters (included with the parameter list so automated convergence testing can change them around)
        straight_length = parameters["straight_length"]
        pml_x_thickness = parameters["pml_x_thickness"]
        pml_y_thickness = parameters["pml_y_thickness"]
        to_pml_x = parameters["to_pml_x"]
        to_pml_y = parameters["to_pml_y"]
        resolution = parameters["resolution"]
        min_run_time = parameters["min_run_time"]
        sigma = parameters["sigma"]
        fields_decay_by = parameters["fields_decay_by"]
        
        # Optimization parameter(s)
        polynomial_coeffs = MaterialFunction.paramsToArray(self.taper_order, 'b', parameters)

        # Simulation cell sizing
        sx = 2 * (pml_x_thickness + to_pml_x) + self.taper_length + 2 * straight_length
        sy = 2 * (to_pml_y + pml_y_thickness) + self.taper_w2
        cell = mp.Vector3(sx, sy)
        
        # Boundary conditions
        boundary_layers = [mp.PML(pml_x_thickness, direction=mp.X),
                           mp.PML(pml_y_thickness, direction=mp.Y)]
        
        # Our device is symmetric, mirror symmetry to speed up computation
        symmetries = [mp.Mirror(mp.Y)]

        # Materials
        Si = mp.Medium(index=Si_index)
        SiO2 = mp.Medium(index=SiO2_index)
        
        # Geometry
        actual_L_straight = (sx - self.taper_length) / 2

        small_wg = mp.Block(mp.Vector3(actual_L_straight, self.taper_w1), center=mp.Vector3(-self.taper_length/2 - actual_L_straight/2, 0), material=Si)
        large_wg = mp.Block(mp.Vector3(actual_L_straight, self.taper_w2), center=mp.Vector3(self.taper_length/2 + actual_L_straight/2, 0), material=Si)
        
        # Sources and such
        input_monitor_pt = mp.Vector3(-0.5 * (self.taper_length + straight_length), 0)
        source_pt = mp.Vector3(-0.5 * self.taper_length - 0.75 * straight_length, 0)
        output_monitor_pt = mp.Vector3(0.5 * (self.taper_length + straight_length), 0)

        sources = [mp.EigenModeSource(src=mp.GaussianSource(1/self.center_wavelength, width=self.gaussian_width),
                                      center=source_pt,
                                      size=mp.Vector3(y=sy-2*pml_y_thickness),
                                      eig_match_freq=True,
                                      eig_parity=mp.ODD_Z+mp.EVEN_Y)]
        
        # Design region setup (using the pdt tools)
        meep_dr_nx = int(resolution * self.taper_length)
        meep_dr_ny = int(resolution * self.taper_w2)
        
        taper = LegendreTaperMaterialFunction(self.taper_order, [self.taper_length, self.taper_w2], self.taper_w1, self.taper_w2)
        design_region = DesignRegion([self.taper_length, self.taper_w2], [meep_dr_nx, meep_dr_ny])
        
        # Design region setup (specific to MEEP)
        meep_design_variables = mp.MaterialGrid(mp.Vector3(meep_dr_nx,meep_dr_ny),SiO2,Si,grid_type='U_MEAN')
        meep_design_region = mpa.DesignRegion(meep_design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(self.taper_length, self.taper_w2, 0)))

        # Material function
        material_function = lambda x: (1.44 + (3.4-1.44)*taper(np.asarray([[x[0]], [x[1]]]), params=MaterialFunction.arrayToParams('b', polynomial_coeffs)))**2
        dr_geometry=mp.Block(size=meep_design_region.size, epsilon_func=material_function)
        
        sim = mp.Simulation(resolution=resolution,
                                 cell_size=cell,
                                 boundary_layers=boundary_layers,
                                 geometry=[small_wg, large_wg, dr_geometry],
                                 sources=sources,
                                 eps_averaging=False, # See what this does
                                 default_material=SiO2,
                                 symmetries=symmetries)
        
        # Input and output monitors
        input_flux = sim.add_flux(1.0/self.center_wavelength, np.max(self.wavelengths) - np.min(self.wavelengths), len(self.wavelengths), mp.FluxRegion(center=input_monitor_pt,size=mp.Vector3(y=sy-2*pml_y_thickness)))
        output_flux = sim.add_flux(1.0/self.center_wavelength, np.max(self.wavelengths) - np.min(self.wavelengths), len(self.wavelengths), mp.FluxRegion(center=output_monitor_pt,size=mp.Vector3(y=sy-2*pml_y_thickness)))
        
        # Run and collect field data
        field_data = []
        collect_fields = lambda mp_sim: None #field_data.append(mp_sim.get_efield_z())
        sim.run(mp.at_every(5, collect_fields), 
                     until_after_sources=mp.stop_when_fields_decayed(min_run_time,mp.Ez,output_monitor_pt,fields_decay_by))
        
                                
        # Collect and process flux data (for the forward direction)
        input_coeffs = sim.get_eigenmode_coefficients(input_flux,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y).alpha[0, :, 0]
        output_coeffs = sim.get_eigenmode_coefficients(output_flux,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y).alpha[0, :, 0]
                    
        transmission = np.mean(np.abs(output_coeffs)**2 / np.abs(input_coeffs)**2)
        
        '''
        # Render
        render = Render("{working_dir}/fields.gif".format(working_dir=self.working_dir))
        max_field = np.max(np.abs(np.real(field_data)))
        for data in field_data:
            fig = plt.figure()
            plt.imshow(sim.get_epsilon().transpose(), interpolation='spline36', cmap='Greys')
            plt.imshow(np.real(data).transpose(), vmin=-max_field, vmax=max_field, interpolation='spline36', cmap='RdBu', alpha=0.9)
            render.add(fig)
            plt.close()
        render.render(10)
        '''
        # Done
        sim.reset_meep()
        
        
        #return Result(parameters, transmission=transmission, field_data=np.asarray(field_data))
        return Result(parameters, transmission=transmission)
            
    
    def process(self, result, parameters):       
        return result # We return this just to assert that we actually do want this data saved  
      
if __name__ == "__main__":
    taper_order = 2
    
    sim = LegendreTaperSimulation(center_wavelength=1.55,
                                  gaussian_width=50,
                                  wavelengths=np.asarray([1.55]),#np.linspace(1.50, 1.60, 3),
                                  taper_order=taper_order, 
                                  taper_w1=.4, 
                                  taper_w2=1, 
                                  taper_length=2)
    
    # Test run to make sure everything is ok
    parameters = {
        "straight_length" : 3,
        "pml_x_thickness" : 1,
        "pml_y_thickness" : 1,
        "to_pml_x" : 1,
        "to_pml_y" : 1,
        "resolution" : 32,
        "min_run_time" : 300,
        "fields_decay_by" : 1e-9,
        "sigma" : 0,
        "b0" : 0.5,
        "b1" : 0,
    }
        
    taper = LegendreTaperMaterialFunction(taper_order, [sim.taper_length, sim.taper_w2], sim.taper_w1, sim.taper_w2)
    design_region = DesignRegion([sim.taper_length, sim.taper_w2], [int(parameters["resolution"] * sim.taper_length), int(parameters["resolution"] * sim.taper_w2)])    
    
    optimizer = ScipyGradientOptimizer(sim, design_region, taper, "transmission", ["b0", "b1"], strategy="maximize")
    optimizer.optimize(parameters, options={"ftol": 0.1})
    
    
    