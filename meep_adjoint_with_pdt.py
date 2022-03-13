#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:18:16 2022

@author: skenning
"""

from pdt.core import Simulation, Util, ParameterChangelog
from pdt.opt import DesignRegion, MaterialFunction, LegendreTaperMaterialFunction

import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa

import numpy as np

import matplotlib.pyplot as plt

import copy

class LegendreTaperSimulation(Simulation):
    def __init__(self, 
                 center_wavelength,
                 gaussian_width,
                 wavelengths,
                 taper_order, 
                 taper_w1, 
                 taper_w2, 
                 taper_length):
        Simulation.__init__(self, "LegendreTaperSimulation", working_dir="legendretapersimulation_working_dir")
        
        # Taper parameters
        self.taper_order = taper_order
        self.taper_w1 = taper_w1
        self.taper_w2 = taper_w2
        self.taper_length = taper_length
        
        # Other simulation parameters
        self.center_wavelength = center_wavelength
        self.gaussian_width = gaussian_width
        self.wavelengths = wavelengths 
        
        # Globally held/updated simulation objects
        self.parameter_changelog = ParameterChangelog()
        self.sim = None
        self.opt = None
        self.taper = None
        self.design_region = None
        
    def run(self, parameters):
        # Convergence parameters (included with the parameter list so automated convergence testing can change them around)
        straight_length = parameters["straight_length"]
        pml_x_thickness = parameters["pml_x_thickness"]
        pml_y_thickness = parameters["pml_y_thickness"]
        to_pml_x = parameters["to_pml_x"]
        to_pml_y = parameters["to_pml_y"]
        resolution = parameters["resolution"]
        
        # Optimization parameter(s)
        polynomial_coeffs = parameters["polynomial_coeffs"]
        
        # We now use the changelog class to see if any of the convergence parameters changed
        self.parameter_changelog.updateParameters(parameters)
        if self.parameter_changelog.changesExclude(["polynomial_coeffs"]): # Rebuild the simulation
            self._log_info("Convergence parameters changed, rebuilding simulation")
            if self.sim: 
                # If other simulations have been ran, we reset meep and all other simulation objects
                self.sim.reset_meep()
                
                self.sim = None
                self.opt = None
                self.taper = None
                self.design_region = None
            
            # Reconstruct everything
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
            Si = mp.Medium(index=3.4)
            SiO2 = mp.Medium(index=1.44)
            
            # Geometry
            actual_L_straight = (sx - self.taper_length) / 2

            small_wg = mp.Block(mp.Vector3(actual_L_straight, self.taper_w1), center=mp.Vector3(-self.taper_length/2 - actual_L_straight/2, 0), material=Si)
            large_wg = mp.Block(mp.Vector3(actual_L_straight, self.taper_w2), center=mp.Vector3(self.taper_length/2 + actual_L_straight/2, 0), material=Si)
            
            # Sources and such
            input_monitor_pt = mp.Vector3(-0.5 * (self.taper_length + straight_length), 0)
            source_pt = mp.Vector3(-0.5 * self.taper_length - 0.75 *  + straight_length, 0)
            output_monitor_pt = mp.Vector3(0.5 * (self.taper_length + straight_length), 0)

            sources = [mp.EigenModeSource(src=mp.GaussianSource(1/self.center_wavelength, width=self.gaussian_width),
                                          center=source_pt,
                                          size=mp.Vector3(y=sy-2*pml_y_thickness),
                                          eig_match_freq=True,
                                          eig_parity=mp.ODD_Z+mp.EVEN_Y)]
            
            # Design region setup (using the pdt tools)
            meep_dr_nx = int(resolution * self.taper_length)
            meep_dr_ny = int(resolution * self.taper_w2)
            
            self.taper = LegendreTaperMaterialFunction(self.taper_order, [self.taper_length, self.taper_w2], self.taper_w1, self.taper_w2)
            self.design_region = DesignRegion([self.taper_length, self.taper_w2], [meep_dr_nx, meep_dr_ny])
            
            # Design region setup (specific to MEEP)
            meep_design_variables = mp.MaterialGrid(mp.Vector3(meep_dr_nx,meep_dr_ny),SiO2,Si,grid_type='U_MEAN')
            meep_design_region = mpa.DesignRegion(meep_design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(self.taper_length, self.taper_w2, 0)))
            
            dr_geometry=mp.Block(size=meep_design_region.size, material=meep_design_variables)
            
            self.sim = mp.Simulation(resolution=resolution,
                                     cell_size=cell,
                                     boundary_layers=boundary_layers,
                                     geometry=[small_wg, large_wg, dr_geometry],
                                     sources=sources,
                                     eps_averaging=False, # See what this does
                                     default_material=SiO2)
            
            # Adjoint monitors
            TE0_input = mpa.EigenmodeCoefficient(self.sim, mp.Volume(center=input_monitor_pt, size=mp.Vector3(y=sy-2*pml_y_thickness)), mode=1, forward=True)
            TE0_output = mpa.EigenmodeCoefficient(self.sim, mp.Volume(center=output_monitor_pt, size=mp.Vector3(y=sy-2*pml_y_thickness)), mode=1, forward=True)
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
            self.opt = mpa.OptimizationProblem(simulation=self.sim, 
                                               objective_functions=objective_function,
                                               objective_arguments=ob_list,
                                               design_regions=[meep_design_region],
                                               frequencies=1/np.asarray(self.wavelengths),
                                               decay_by=1e-5,
                                               minimum_run_time=300)
            
            # End reconstruction of the simulation
        else:
            self._log_info("Convergence parameters did not change")
            
        # Now we update the design region (meep's) and give it a nice plot
        design = self.design_region.evalMaterialFunction(self.taper, MaterialFunction.arrayToParams('b', polynomial_coeffs))
        self.opt.update_design([design.transpose().flatten()])

        plt.figure()
        self.opt.plot2D(True)
        
        # Run the forward and adjoint run
        f0, dJ_du = (self.opt)()
        
        return dJ_du, self.wavelengths, self.design_region.N

def plot_gradient(dJ_du, wavelengths, N):
    if Util.is_master():
        if len(dJ_du.shape) > 1:
            plt.figure()
            #print(dJ_du)
            plt.imshow(npa.sum(dJ_du, axis=1).reshape(N).transpose())
            '''
            print("dJ_du.shape", dJ_du.shape)
            for i in range(len(wavelengths)):
                plt.figure()
                plt.title("{wavelength}".format(wavelength=round(wavelengths[i], 2)))
                plt.imshow(dJ_du[:,i].reshape(N).transpose())
            '''
        else:
            plt.figure()
            plt.imshow(dJ_du.reshape(N).transpose())
            plt.savefig("result.png")

if __name__ == "__main__":
    sim = LegendreTaperSimulation(center_wavelength=1.55,
                                  gaussian_width=50,
                                  wavelengths=np.linspace(1.50, 1.60, 3),
                                  taper_order=10, 
                                  taper_w1=1, 
                                  taper_w2=5, 
                                  taper_length=5)
    
    # Test run to make sure everything is ok
    test_parameters = {
        "straight_length" : 5,
        "pml_x_thickness" : 3,
        "pml_y_thickness" : 3,
        "to_pml_x" : 3,
        "to_pml_y" : 3,
        "resolution" : 20,
        "polynomial_coeffs" : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    
    dJ_du, wavelengths, N = sim.run(test_parameters)
    plot_gradient(dJ_du, wavelengths, N)
    
    test_parameters["polynomial_coeffs"] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dJ_du, wavelengths, N = sim.run(test_parameters)
    plot_gradient(dJ_du, wavelengths, N)
    
    test_parameters["polynomial_coeffs"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    dJ_du, wavelengths, N = sim.run(test_parameters)
    plot_gradient(dJ_du, wavelengths, N)

