#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:18:16 2022

@author: skenning
"""

from pdt.core import Simulation, Util, ParameterChangelog, Result, PreviousResults, ConvergenceTest
from pdt.opt import DesignRegion, MaterialFunction, LegendreTaperMaterialFunction, ScipyGradientOptimizer
from pdt.tools import Render, ProgressRender

import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import h5py as h5

class LegendreTaperSimulation(Simulation):
    def __init__(self, 
                 center_wavelength,
                 gaussian_width,
                 wavelengths,
                 taper_order, 
                 taper_w1, 
                 taper_w2, 
                 taper_length,
                 catch_errors=False,
                 render=False):
        Simulation.__init__(self, "WaveguideTaperAdjvsFD", working_dir="WD_WaveguideTaperAdjvsFD", catch_errors=catch_errors)
        
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
        
        # Debug and renders
        self.render = render
        
    def getCurrentDesign(self):
        return self.taper
    def getCurrentDesignRegion(self):
        return self.design_region

    def run(self, parameters):
        # Some constants
        Si_index = 3.4
        SiO2_index = 1.44
        parity = mp.ODD_Y + mp.EVEN_Z
        
        # Convergence parameters (included with the parameter list so automated convergence testing can change them around)
        straight_length = float(parameters["straight_length"])
        pml_x_thickness = float(parameters["pml_x_thickness"])
        pml_y_thickness = float(parameters["pml_y_thickness"])
        to_pml_x = float(parameters["to_pml_x"])
        to_pml_y = float(parameters["to_pml_y"])
        resolution = int(parameters["resolution"])
        min_run_time = int(parameters["min_run_time"])
        fields_decay_by = float(parameters["fields_decay_by"])
        include_jac = parameters["include_jac"]
        sigma = 0
                        
        # Optimization parameter(s)
        polynomial_coeffs = MaterialFunction.paramsToArray(self.taper_order, "b", parameters)
        
        # We now use the changelog class to see if any of the convergence parameters changed
        self.parameter_changelog.updateParameters(parameters)
        if self.parameter_changelog.changesExclude(MaterialFunction.paramListHelper(self.taper_order, "b")): # Rebuild the simulation
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
            symmetries = []#[mp.Mirror(mp.Y)]

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
                                          eig_parity=parity)]
            
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
                                     default_material=SiO2,
                                     symmetries=symmetries)
            
            # Adjoint monitors
            TE0_input = mpa.EigenmodeCoefficient(self.sim, mp.Volume(center=input_monitor_pt, size=mp.Vector3(y=sy-2*pml_y_thickness)), mode=1, eig_parity=parity, forward=True)
            TE0_output = mpa.EigenmodeCoefficient(self.sim, mp.Volume(center=output_monitor_pt, size=mp.Vector3(y=sy-2*pml_y_thickness)), mode=1, eig_parity=parity, forward=True)
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
                                               frequencies=1.0/np.asarray(self.wavelengths),
                                               decay_by=1e-9,
                                               minimum_run_time=min_run_time)
            
            # End reconstruction of the simulation
        else:
            self._log_info("Convergence parameters did not change")
        
        # Now we update the design region (meep's) and give it a nice plot
        design = self.design_region.evalMaterialFunction(self.taper, MaterialFunction.arrayToParams('b', polynomial_coeffs), sigma)
        self.opt.update_design([design.transpose().flatten()])
        
        if self.render:
            '''
            # Nice animation
            field_data = []
            collect_fields = lambda mp_sim: field_data.append(mp_sim.get_efield_y())
            self.sim.run(mp.at_every(2, collect_fields), 
                         until_after_sources=mp.stop_when_fields_decayed(min_run_time,mp.Ey,output_monitor_pt,fields_decay_by))
            
            render = Render("{working_dir}/render.gif".format(working_dir=self.working_dir))
            max_field = np.max(np.abs(np.real(field_data)))
            for data in field_data:
                fig = plt.figure(dpi=128)
                plt.imshow(self.sim.get_epsilon().transpose(), interpolation='spline36', cmap='binary', extent=(-sx/2, sx/2, -sy/2, sy/2))
                plt.imshow(np.real(data).transpose(), vmin=-max_field, vmax=max_field, interpolation='spline36', cmap='RdBu', alpha=0.9, extent=(-sx/2, sx/2, -sy/2, sy/2))
                plt.xlabel("x (??m)")
                plt.ylabel("y (??m)")
                render.add(fig)
                plt.close()
            render.render(100)
            '''
            # Plot out derivatives
            self.design_region.plotMaterialFunctionDerivative(self.taper, MaterialFunction.arrayToParams('b', polynomial_coeffs), sigma)
            
            '''
            # Generate a visualization of the sensitivity
            f0, dJ_du = (self.opt)()
            if len(dJ_du.shape) > 1:
                dJ_du = npa.sum(dJ_du, axis=1)
            dJ_du = dJ_du.reshape(self.design_region.N).transpose()
            
            plt.figure()
            plt.imshow(dJ_du, extent=(-self.taper_length/2, self.taper_length/2, -self.taper_w2/2, self.taper_w2/2))
            plt.xlabel("x (??m)")
            plt.ylabel("y (??m)")
            plt.colorbar()
            '''
        else:
            if include_jac:
                # Run the forward and adjoint run
                f0, dJ_du = (self.opt)()
                if len(dJ_du.shape) > 1:
                    dJ_du = npa.sum(dJ_du, axis=1)
                dJ_du = dJ_du.reshape(self.design_region.N).transpose()
                
                # Compute the gradient with respect to the taper parameters
                order, du_db, min_db, all_du_db = self.design_region.evalMaterialFunctionDerivative(self.taper, MaterialFunction.arrayToParams('b', polynomial_coeffs), sigma) # Material function vs taper coefficients
                dJ_db = np.asarray([np.sum(dJ_du * du_db[i]) for i in range(len(order))]).flatten()
            
                self._log_info("adj gradient: {gradient}".format(gradient=dJ_db))
            
                return Result(parameters, f0=f0, dJ_du=dJ_du, dJ_db=dJ_db, min_db=np.asarray(min_db), all_du_db=all_du_db, wavelengths=self.wavelengths)
            else:
                self.opt.forward_run()
                return Result(parameters, f0=self.opt.f0, wavelengths=self.wavelengths)
    
    def process(self, result, parameters):        
        return result # We return this just to assert that we actually do want this data saved  
      
def getBestParameters():
    pr = PreviousResults("WaveguideTaperAdjvsFD", "WD_WaveguideTaperAdjvsFD")
    
    def objective(f0):
        return f0[()] # get the scalar data out of the HDF5 Dataset
    best = pr.getBestParameters("f0", objective, True)
    
    if best:
        print(best)
    else:
        print("no previous simulation data found")
      
if __name__ == "__main__":
    render = False
    convergence_test_first = True
    resolution = 16
    
    taper_order = 10
    
    sim = LegendreTaperSimulation(center_wavelength=1.55,
                                  gaussian_width=5,
                                  wavelengths=np.linspace(1.50, 1.60, 3),
                                  taper_order=taper_order, 
                                  taper_w1=.4, 
                                  taper_w2=1, 
                                  taper_length=1,
                                  render=render)
    
    # Start from the best simulation logged to the HDF5 file
    pr = PreviousResults("WaveguideTaperAdjvsFD", "WD_WaveguideTaperAdjvsFD")
    def pr_objective(f0):
        return f0[()] # Retrive "scalar" data from the HDF5
    previous_best_parameters = pr.getBestParameters("f0", pr_objective, True)
    
    # Simulation start parameters
    parameters = None
    if previous_best_parameters and not render:
        parameters = previous_best_parameters
        sim._log_info("starting with previous run's best parameters: {parameters}".format(parameters=parameters))
    else:
        # Set up the simulation parameters/convergence parameters
        parameters = {
            "straight_length" : 5,
            "pml_x_thickness" : 1,
            "pml_y_thickness" : 1,
            "to_pml_x" : 1,
            "to_pml_y" : 1,
            "min_run_time" : 100,
            "fields_decay_by" : 1e-9,
            "include_jac" : False
        }
        for bi in MaterialFunction.paramListHelper(taper_order, "b"):
            parameters[bi] = 0
    
        parameters["b0"] = 0 # We start with something absurd, and then improve it.
    parameters["resolution"] = resolution
    parameters["include_jac"] = False
        
    if convergence_test_first:
        ct = ConvergenceTest(sim, 
                             convergence_parameters={"resolution" : (16, 128),
                                                     "straight_length" : (1.0, 10.0),
                                                     "pml_x_thickness" : (1.0, 5.0),
                                                     "pml_y_thickness" : (1.0, 5.0),
                                                     "to_pml_x" : (1.0, 5.0),
                                                     "to_pml_y" : (1.0, 5.0)},
                             fom="f0",
                             relative_change=0.01,
                             absolute_change=0.01,
                             max_steps=10,
                             certainty_steps=3,
                             interval_generation=np.linspace)
        '''
        The convergence_parameters dict is noteworthy to talk about. If the tuple
        passed for the bounds consists of floats (i.e., there is a decimal place 
        in both numbers, indicating their floating point status), then the code will not
        round to the nearest integer. Otherwise, it will round to the nearest integer.
        For example, since the bounds for resolution are both ints, the code will round
        the sweeped points to the nearest integer.        
        '''
        
        parameters = ct.run(parameters,
                            plot_fname="convergence")
    
    
    if render:
        sim.oneOff(parameters)
    else:
        # Optimizer setup   
        scigro = ScipyGradientOptimizer(sim, 
                                        sim.getCurrentDesignRegion, 
                                        sim.getCurrentDesign, 
                                        method="L-BFGS-B",
                                        fom="f0", 
                                        jac="dJ_db", 
                                        opt_parameters=MaterialFunction.paramListHelper(taper_order, "b"), 
                                        strategy="maximize")
        
        scigro.optimize(parameters, 
                        finite_difference=False,
                        progress_render_fname="progress.gif", 
                        progress_render_fig_kwargs=dict(figsize=(10, 15)), 
                        progress_render_duration=1000, 
                        bounds=[(0, 1)]*taper_order, 
                        options=dict(maxls=10, disp=True))
    
    

    
        