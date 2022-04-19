#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:47:42 2022

@author: skenning
"""

from pdt.core import Simulation, Util, ParameterChangelog, Result, PreviousResults
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

# %%% Material Function Setup %%%
from scipy.special import legendre
class DengSymMMIStripToSlot(MaterialFunction):
    def __init__(self, 
                 taper_order,
                 strip_width,
                 slot_width,
                 slot_rail_width,
                 device_length,
                 mat_height,
                 ridge_order,
                 ridge_height,
                 max_width,
                 mmi_width=None,
                 mmi_length=None,
                 input_ridge_runup=None):
        
        # Placeholders for now
        db_hints = {
            "mmi_width" : (.01, device_length, 1000),
            "mmi_length" : (.01, device_length, 1000),
            "input_ridge_runup" : (.01, device_length, 1000)
            }
        db_hints.update(MaterialFunction.hintHelper(taper_order, "taper", (.1, 1, 10)))
        db_hints.update(MaterialFunction.hintHelper(ridge_order, "ridge", (.1, 1, 10)))
        MaterialFunction.__init__(self, db_hints=db_hints)
        
        # Primary values
        self.taper_order = taper_order
        self.strip_width = strip_width
        self.slot_width = slot_width
        self.slot_rail_width = slot_rail_width
        self.device_length = device_length
        self.mat_height = mat_height
        self.ridge_order = ridge_order
        self.ridge_height = ridge_height
        self.max_width = max_width
        
        self.mmi_width = mmi_width
        self.mmi_length = mmi_length
        self.input_ridge_runup = input_ridge_runup
        
        # Basis for the taper
        self.taper_basis = [legendre(2*n + 1) for n in range(self.taper_order)] 
        self.ridge_basis = [legendre(2*n + 1) for n in range(self.ridge_order)] 
        
    def getMaxDimensions(self):
        return (self.device_length, self.max_width)
        
    def evalModel(self, coords, params: dict[str, float]):        
        # Drawing setup
        val = np.zeros(coords[0].shape)
        
        if self.mmi_width:
            mmi_width = self.mmi_width
        else:
            mmi_width = params["mmi_width"]
        if self.mmi_length:
            mmi_length = self.mmi_length
        else:
            mmi_length = params["mmi_length"]
        if self.input_ridge_runup:
            input_ridge_runup = self.input_ridge_runup
        else:
            input_ridge_runup = params["input_ridge_runup"]
                
        # Polynomial coefficients for our design
        taper_coeff = MaterialFunction.paramsToArray(self.taper_order, "taper", params)
        ridge_coeff = MaterialFunction.paramsToArray(self.ridge_order, "ridge", params)
           
        # Convenience variables
        x, y = coords
        
        left = -self.device_length/2
        right = self.device_length/2
        
        ### We draw the basic device without striploading, starting from the left side ###        
        # Masks to distinguish regions
        runup_mask = (x < (left + input_ridge_runup))
        mmi_mask = ((~(runup_mask)) & (x < left + input_ridge_runup + mmi_length)) 
        taper_mask = (~runup_mask) & (~mmi_mask)
        slot_mask = taper_mask & (np.abs(y) < self.slot_width / 2)
        
        # Now the taper. We'll use polynomials for it's outer boundary for fun and an extra parameter to tune.
        travel_x = self.device_length - mmi_length - input_ridge_runup
        travel_y = (mmi_width - (self.slot_width + 2 * self.slot_rail_width)) / 2
        
        def taper_w(x):
            s = -2 * (((x - (left + mmi_length + input_ridge_runup)) / travel_x) - 0.5)            
            w = np.zeros(x.shape)
            
            for i, poly in enumerate(self.taper_basis):
                w += taper_coeff[i] * ((poly(s) + 1) / 2) * travel_y 
            w += (self.slot_width + 2 * self.slot_rail_width) / 2
            w[np.abs(s) > 1] = 0
            return w
        
        # Full height mask consists of (runup | mmi section | taper)
        full_height_mask = (runup_mask & (np.abs(y) < self.strip_width/2)) | (mmi_mask & (np.abs(y) < mmi_width/2)) | (~(slot_mask) & taper_mask & (np.abs(y) < taper_w(x)))
        val[full_height_mask] = 1
        
        # Now we work on the ridge area
        ridge_mask = ~full_height_mask & ~slot_mask
        effective_index = self.ridge_height / self.mat_height
        
        def ridge_w(x):
            s = x / right
            w = np.zeros(x.shape)
            
            for i, poly in enumerate(self.ridge_basis):
                w += ridge_coeff[i] * ((poly(s) + 1) / 2) * (self.max_width / 2 - self.strip_width / 2)
            w += self.strip_width / 2
            return w
        
        val[ridge_mask & (np.abs(y) < ridge_w(x))] = effective_index
        
        return val
    
# %%% Simulation setup %%% 
class DengSymMMIStripToSlotSimulation(Simulation):
    def __init__(self,
                 fname="DengMMIStripToSlot",
                 taper_order=1,
                 ridge_order=1,
                 strip_width=0.400,
                 slot_width=0.100,
                 slot_rail_width=0.150,
                 mat_height=0.250,
                 ridge_height=0.040,
                 center_wavelength=1.55,
                 wavelengths=np.linspace(1.50, 1.60, 3),
                 gaussian_width=10,
                 opt_parameters=[],
                 catch_errors=False,
                 render=False):
        Simulation.__init__(self, fname, working_dir="WD_DengMMIStripToSlot", catch_errors=catch_errors)

        # Order of boundaries
        self.taper_order = taper_order
        self.ridge_order = ridge_order

        # Known parameters, based off of the platform
        self.strip_width = strip_width
        self.slot_width = slot_width
        self.slot_rail_width = slot_rail_width
        self.mat_height = mat_height
        self.ridge_height = ridge_height
        
        # Sources
        self.center_wavelength = center_wavelength
        self.wavelengths = wavelengths
        self.gaussian_width = gaussian_width
        
        # Debug stuff
        self.catch_errors = catch_errors
        self.render = render
        
        # Gradient evaluation
        self.opt_parameters = opt_parameters
        
        # Globally held/updated simulation objects
        self.parameter_changelog = ParameterChangelog()
        self.sim = None
        self.opt = None
        self.converter = None
        self.design_region = None
        
    def getCurrentDesign(self):
        return self.converter
    def getCurrentDesignRegion(self):
        return self.design_region
        
    def run(self, parameters):
        # Some constants
        Si_index = 3.4
        SiO2_index = 1.44
        weighted_index = np.sqrt((self.ridge_height * Si_index**2 + (self.mat_height - self.ridge_height) * SiO2_index**2) / self.mat_height)
        parity = mp.ODD_Y + mp.EVEN_Z
        
        # Convergence parameters (included with the parameter list so automated convergence testing can change them around)
        straight_length = float(parameters["straight_length"])
        pml_x_thickness = float(parameters["pml_x_thickness"])
        pml_y_thickness = float(parameters["pml_y_thickness"])
        to_pml_x = float(parameters["to_pml_x"])
        to_pml_y = float(parameters["to_pml_y"])
        resolution = int(parameters["resolution"])
        min_run_time = int(parameters["min_run_time"])
        device_length = float(parameters["device_length"])
        device_width = float(parameters["device_width"])
        min_run_time = float(parameters["min_run_time"])
        fields_decay_by = float(parameters["fields_decay_by"])
        include_jac = bool(parameters["include_jac"])
        sigma = 0
        
        # Geometry parameters
        mmi_length = float(parameters["mmi_length"])
        mmi_width = float(parameters["mmi_width"])
        input_ridge_runup = float(parameters["input_ridge_runup"])
        
        # Parameters expected to be changed around in the design region        
        opt_parameter_dict = dict((k,v) for k, v in parameters.items() if k in opt_parameters)
        
        self.parameter_changelog.updateParameters(parameters)
        if self.parameter_changelog.changesExclude(opt_parameters): # Rebuild the simulation
            self._log_info("Convergence parameters changed, rebuilding simulation")
            if self.sim: 
                # If other simulations have been ran, we reset meep and all other simulation objects
                self.sim.reset_meep()
                
                self.sim = None
                self.opt = None
                self.taper = None
                self.design_region = None
        
            # Reconstruct everything
            sx = 2 * (pml_x_thickness + to_pml_x) + device_length + 2 * straight_length
            sy = 2 * (to_pml_y + pml_y_thickness) + device_width
            cell = mp.Vector3(sx, sy)
            
            # Boundary conditions
            boundary_layers = [mp.PML(pml_x_thickness, direction=mp.X),
                               mp.PML(pml_y_thickness, direction=mp.Y)]
            
            # Our device is symmetric, mirror symmetry to speed up computation
            symmetries = []
    
            # Materials
            Si = mp.Medium(index=Si_index)
            SiO2 = mp.Medium(index=SiO2_index)
            meanSiSiO2 = mp.Medium(index=weighted_index)
            
            # Geometry
            actual_L_straight = (sx - device_length) / 2
    
            # Sources and such            
            input_monitor_pt = mp.Vector3(-0.5 * (device_length + straight_length), 0)
            source_pt = mp.Vector3(-0.5 * device_length - 0.75 * straight_length, 0)
            output_monitor_pt = mp.Vector3(0.5 * (device_length + straight_length), 0)
            sources = [mp.EigenModeSource(src=mp.GaussianSource(1/self.center_wavelength, width=self.gaussian_width),
                                          center=source_pt,
                                          size=mp.Vector3(y=sy-2*pml_y_thickness),
                                          eig_match_freq=True,
                                          eig_parity=parity)]
    
            # Design region setup (using the pdt tools)
            meep_dr_nx = int(resolution * device_length)
            meep_dr_ny = int(resolution * device_width)
            
            self.converter = DengSymMMIStripToSlot(taper_order=self.taper_order,
                                                   strip_width=self.strip_width,
                                                   slot_width=self.slot_width,
                                                   slot_rail_width=self.slot_rail_width,
                                                   device_length=device_length,
                                                   mat_height=self.mat_height,
                                                   ridge_order=self.ridge_order,
                                                   ridge_height=self.ridge_height,
                                                   max_width=device_width,
                                                   mmi_width=mmi_width,
                                                   mmi_length=mmi_length,
                                                   input_ridge_runup=input_ridge_runup)
            self.design_region = DesignRegion([device_length, device_width], [meep_dr_nx, meep_dr_ny])
            
            # Design region setup (specific to MEEP)
            meep_design_variables = mp.MaterialGrid(mp.Vector3(meep_dr_nx,meep_dr_ny),SiO2,Si,grid_type="U_MEAN")
            meep_design_region = mpa.DesignRegion(meep_design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(device_length, device_width, 0)))
    
            dr_geometry=mp.Block(size=meep_design_region.size, material=meep_design_variables)
            input_wg = mp.Block(mp.Vector3(actual_L_straight, self.strip_width), center=mp.Vector3(-device_length/2 - actual_L_straight/2, 0), material=Si)
            output_rail_top = mp.Block(mp.Vector3(actual_L_straight, self.slot_rail_width), center=mp.Vector3(device_length/2 + actual_L_straight/2, (self.slot_width + self.slot_rail_width) / 2), material=Si)
            output_rail_bottom = mp.Block(mp.Vector3(actual_L_straight, self.slot_rail_width), center=mp.Vector3(device_length/2 + actual_L_straight/2, -(self.slot_width + self.slot_rail_width) / 2), material=Si)
            
            ridge_center_x = device_length/2 + actual_L_straight/2
            ridge_center_y = (sy / 4) + self.slot_rail_width + self.slot_width / 2
            ridge_top = mp.Block(mp.Vector3(actual_L_straight, sy / 2), center=mp.Vector3(ridge_center_x, ridge_center_y), material=meanSiSiO2)
            ridge_bottom = mp.Block(mp.Vector3(actual_L_straight, sy / 2), center=mp.Vector3(ridge_center_x, -ridge_center_y), material=meanSiSiO2)

            self.sim = mp.Simulation(resolution=resolution,
                                     cell_size=cell,
                                     boundary_layers=boundary_layers,
                                     geometry=[input_wg, dr_geometry, output_rail_top, output_rail_bottom, ridge_top, ridge_bottom],
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
            
        else:
            self._log_info("Convergence parameters did not change")        
                
        # Now we update the design region (meep's) and give it a nice plot
        design = self.design_region.evalMaterialFunction(self.converter, opt_parameter_dict, sigma)
        self.opt.update_design([design.transpose().flatten()])
        
        if self.render:
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
                plt.xlabel("x (µm)")
                plt.ylabel("y (µm)")
                render.add(fig)
                plt.close()
            render.render(50)
        else:
            if include_jac:
                # Run the forward and adjoint run
                f0, dJ_du = (self.opt)()
                if len(dJ_du.shape) > 1:
                    dJ_du = npa.sum(dJ_du, axis=1)
                dJ_du = dJ_du.reshape(self.design_region.N).transpose()
                
                plt.figure()
                plt.imshow(dJ_du)
                plt.colorbar()
                
                # Compute the gradient with respect to the taper parameters
                order, du_db, min_db, all_du_db = self.design_region.evalMaterialFunctionDerivative(self.converter, opt_parameter_dict, sigma) # Material function vs taper coefficients
                dJ_db = np.asarray([np.sum(dJ_du * du_db[i]) for i in range(len(order))]).flatten()
                
                '''
                all_dJ_db = []
                for i in range(len(order)):
                    all_dJ_db.append(np.asarray([np.sum(dJ_du * all_du_db[i][j]) for j in range(len(order))]).flatten())
                all_dJ_db = np.asarray(all_dJ_db)
                '''
                
                self._log_info("adj. gradient: {gradient}".format(gradient=dJ_db))
                
                return Result(parameters, f0=f0, dJ_du=dJ_du, dJ_db=dJ_db*10, min_db=np.asarray(min_db), all_du_db=all_du_db, wavelengths=self.wavelengths)
            else:
                self.opt.forward_run()
                return Result(parameters, f0=self.opt.f0)
        
    def process(self, result, parameters):
        return result

def getBestParameters():
    pr = PreviousResults("DengMMIStripToSlot", "WD_DengMMIStripToSlot")
    
    def objective(f0):
        return f0[()] # get the scalar data out of the HDF5 Dataset
    best = pr.getBestParameters("f0", objective, True)
    
    if best:
        print(best)
    else:
        print("no previous simulation data found")
        
    return best

if __name__ == "__main__":
    render = False
    resolution = 64
    
    # Constraints
    ridge_order = 6
    taper_order = 6

    device_length = 6
    device_width = 3
    
    # Starting stuff
    mmi_length_start = 1.38
    mmi_width_start = 1.24
    input_ridge_runup_start = 1

    # Parameters to optimize over
    opt_parameters = [] #["mmi_length", "mmi_width", "input_ridge_runup"]
    opt_parameters.extend(MaterialFunction.paramListHelper(taper_order, "taper"))
    opt_parameters.extend(MaterialFunction.paramListHelper(ridge_order, "ridge"))

    # Simulate our design    
    sim = DengSymMMIStripToSlotSimulation(fname="DengMMIStripToSlot",
                                          taper_order=taper_order,
                                          ridge_order=ridge_order,
                                          opt_parameters=opt_parameters,
                                          render=render)
    
    previous_best = getBestParameters()
    
    parameters = None
    if previous_best and not render:
        parameters = previous_best
        sim._log_info("starting with the previous best of {parameters}".format(parameters=parameters))
    else:
        parameters = {
            "mmi_length" : mmi_length_start,
            "mmi_width_start" : mmi_width_start,
            "input_ridge_runup" : input_ridge_runup_start,
            "straight_length" : 3,
            "pml_x_thickness" : 1,
            "pml_y_thickness" : 1,
            "to_pml_x" : 1,
            "to_pml_y" : 1,
            "min_run_time" : 100,
            "fields_decay_by" : 1e-9,
            "device_length" : device_length,
            "device_width" : device_width,
            "mmi_length" : mmi_length_start,
            "mmi_width" : mmi_width_start,
            "input_ridge_runup" : input_ridge_runup_start,
            "include_jac" : False
        }            
        for i, taper_i in enumerate(MaterialFunction.paramListHelper(taper_order, "taper")):
            if i == 0:
                parameters[taper_i] = 1
            else:
                parameters[taper_i] = 0
        for i, ridge_i in enumerate(MaterialFunction.paramListHelper(taper_order, "ridge")):
            if i == 0:
                parameters[ridge_i] = 0.25
            else:
                parameters[ridge_i] = 0
        
    parameters["resolution"] = resolution
    if render:
        sim.oneOff(parameters)
    else:
        optimizer = ScipyGradientOptimizer(sim, 
                                           sim.getCurrentDesignRegion, 
                                           sim.getCurrentDesign, 
                                           "f0", 
                                           "dJ_db", 
                                           opt_parameters, 
                                           strategy="maximize",
                                           method="L-BFGS-B")
        
        #bounds = [(mmi_length_start,mmi_length_start), (mmi_width_start,mmi_width_start), (input_ridge_runup_start,input_ridge_runup_start)]
        #bounds.extend([(0, 1)]*(taper_order+ridge_order))
        optimizer.optimize(parameters, 
                           finite_difference=False,
                           progress_render_fname="progress.gif", 
                           progress_render_fig_kwargs=dict(figsize=(10, 15)), 
                           progress_render_duration=1000, 
                           #bounds=bounds,
                           options=dict(maxls=10, maxfun=10, maxiter=10))
