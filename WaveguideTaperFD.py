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
        Simulation.__init__(self, "LegendreTaperSimulation", working_dir="lts_working_dir", catch_errors=catch_errors)
        
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
        polynomial_coeffs = parameters["polynomial_coeffs"]
        
        # We now use the changelog class to see if any of the convergence parameters changed
        self.parameter_changelog.updateParameters(parameters)
        if (not self.optimize) or self.parameter_changelog.changesExclude(["polynomial_coeffs", "sigma", "min_run_time", "fields_decay_by"]): # Rebuild the simulation
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
            
            self.taper = LegendreTaperMaterialFunction(self.taper_order, [self.taper_length, self.taper_w2], self.taper_w1, self.taper_w2)
            self.design_region = DesignRegion([self.taper_length, self.taper_w2], [meep_dr_nx, meep_dr_ny])
            
            # Design region setup (specific to MEEP)
            meep_design_variables = mp.MaterialGrid(mp.Vector3(meep_dr_nx,meep_dr_ny),SiO2,Si,grid_type='U_MEAN')
            meep_design_region = mpa.DesignRegion(meep_design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(self.taper_length, self.taper_w2, 0)))

            # Material function
            material_function = lambda x: (1.44 + (3.4-1.44)*self.taper(np.asarray([[x[0]], [x[1]]]), params=MaterialFunction.arrayToParams('b', polynomial_coeffs)))**2
            dr_geometry=mp.Block(size=meep_design_region.size, epsilon_func=material_function)
            
            self.sim = mp.Simulation(resolution=resolution,
                                     cell_size=cell,
                                     boundary_layers=boundary_layers,
                                     geometry=[small_wg, large_wg, dr_geometry],
                                     sources=sources,
                                     eps_averaging=False, # See what this does
                                     default_material=SiO2,
                                     symmetries=symmetries)
            
        else:
            self._log_info("Convergence parameters did not change")
        
        
        # Input and output monitors
        input_flux = self.sim.add_flux(1.0/self.center_wavelength, np.max(self.wavelengths) - np.min(self.wavelengths), len(self.wavelengths), mp.FluxRegion(center=input_monitor_pt,size=mp.Vector3(y=sy-2*pml_y_thickness)))
        output_flux = self.sim.add_flux(1.0/self.center_wavelength, np.max(self.wavelengths) - np.min(self.wavelengths), len(self.wavelengths), mp.FluxRegion(center=output_monitor_pt,size=mp.Vector3(y=sy-2*pml_y_thickness)))
        
        # Show it
        plt.figure()
        self.sim.plot2D()
        plt.savefig("lts_working_dir/progress_device.png")
        
        # Run and collect field data
        
        field_data = []
        collect_fields = lambda mp_sim: field_data.append(mp_sim.get_efield_z())
        self.sim.run(mp.at_every(5, collect_fields), 
                     until_after_sources=mp.stop_when_fields_decayed(min_run_time,mp.Ez,output_monitor_pt,fields_decay_by))
        
                                
        # Collect and process flux data (for the forward direction)
        input_coeffs = self.sim.get_eigenmode_coefficients(input_flux,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y).alpha[0, :, 0]
        output_coeffs = self.sim.get_eigenmode_coefficients(output_flux,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y).alpha[0, :, 0]
                    
        transmission = np.abs(output_coeffs)**2 / np.abs(input_coeffs)**2
        
        
        # Render
        render = Render("lts_working_dir/fields.gif")
        max_field = np.max(np.abs(np.real(field_data)))
        for data in field_data:
            fig = plt.figure()
            plt.imshow(self.sim.get_epsilon().transpose(), interpolation='spline36', cmap='Greys')
            plt.imshow(np.real(data).transpose(), vmin=-max_field, vmax=max_field, interpolation='spline36', cmap='RdBu', alpha=0.9)
            render.add(fig)
            plt.close()
        render.render(10)
        
        
        #return Result(parameters, transmission=transmission, field_data=np.asarray(field_data))
        return Result(parameters, transmission=transmission)
            
    
    def process(self, result, parameters):       
        return result # We return this just to assert that we actually do want this data saved  
      
if __name__ == "__main__":
    taper_order = 1
    
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
        "resolution" : 20,
        "min_run_time" : 300,
        "fields_decay_by" : 1e-9,
        "sigma" : 0
    }
    

    ms_optimizer = MinStepOptimizer([0.5]*taper_order)
    def g(x):
        parameters["polynomial_coeffs"] = tuple(x)
        result = sim.oneOff(parameters)
        
        f0, jacobian, min_db = (result.values["f0"], result.values["dJ_db"], result.values["min_db"])
        sim._log_info("optimizer visited {x}: {f0}, {jacobian}".format(x=x, f0=f0, jacobian=jacobian))
                
        return (result.values["f0"], result.values["dJ_db"], result.values["min_db"])
    
    
    x0 = [0]*taper_order
    x0[0] = 1
    if sim.optimize:
        #scipy.optimize.minimize(f, x0, method='CG', jac=jac, options={'gtol': 1e-10})
        #ms_optimizer.maximize(g, x0)
        
        parameters["polynomial_coeffs"] = tuple(x0)
        sim.oneOff(parameters)
        
        '''
        prev_f0 = None
        prev_w2 = None
        for w2 in np.linspace(.3, 1, 10):
            x0[0] = w2
            parameters["polynomial_coeffs"] = tuple(x0)
            
            result = sim.oneOff(parameters)
            f0 = result.values['f0']
            jacobian = np.mean(result.values["dJ_db"])
            
            if prev_f0 and prev_w2:
                sim._log_info("visited {w2}: f0={f0}, df0/dw2={df0_dw2}, jac={jacobian}".format(w2=w2, f0=f0, jacobian=jacobian, df0_dw2=(f0-prev_f0)/(w2-prev_w2)))
            else:
                sim._log_info("visited {w2}: f0={f0}, jac={jacobian}".format(w2=w2, f0=f0, jacobian=jacobian))
                
            prev_f0 = f0
            prev_w2 = w2
        '''
            
    else:
        parameters["polynomial_coeffs"] = tuple(x0)
        sim.oneOff(parameters)
        
        '''
        for w2 in np.linspace(0, 1, 100):
            x0[0] = w2
            parameters["polynomial_coeffs"] = tuple(x0)
            
            sim.oneOff(parameters)
        '''  
        