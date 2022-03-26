#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:47:42 2022

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

# %%% Material Function Setup %%%
class DengSymMMIStripToSlot(MaterialFunction):
    def __init__(self, 
                 strip_width,
                 slot_width,
                 slot_rail_width,
                 device_length,
                 mat_height,
                 ridge_height):
        
        # Placeholders for now
        db_hints = {
            "mmi_width" : (0, device_length, 1000),
            "mmi_length" : (0, device_length, 1000)
            }
        MaterialFunction.__init__(self, db_hints=db_hints)
        
        # Primary values
        self.strip_width = strip_width
        #self.mmi_width = mmi_width
        #self.mmi_length = mmi_length
        self.slot_width = slot_width
        self.slot_rail_width = slot_rail_width
        self.device_length = device_length
        self.mat_height = mat_height
        self.ridge_height = ridge_height
        
        
    def evalModel(self, x, params: dict[str, float]):        
        # Drawing setup
        val = np.zeros(x[0].shape)
        mmi_width = params["mmi_width"]
        mmi_length = params["mmi_length"]
           
        ### We draw the basic device without striploading, starting from the left side ###        
        # Actual converter mask
        converter_mask = np.abs(x[0]) < (self.device_length / 2)
        
        # For MMI section
        val[(np.abs(x[1]) < mmi_width/2) & (x[0] < (mmi_length - self.device_length / 2)) & converter_mask] = 1
        
        # Slot mask
        slot_mask = (np.abs(x[1]) < self.slot_width/2) & (x[0] >= (mmi_length - self.device_length / 2))
        
        # Taper
        taper_length = self.device_length - mmi_length
        taper_end_width = self.slot_rail_width * 2 + self.slot_width
        taper_travel = mmi_width / 2 - taper_end_width / 2
        
        def taper_height(x):
            slope = taper_travel / taper_length
            x_shift = x - (mmi_length - self.device_length / 2)
            return (mmi_width / 2) - x_shift * slope
        
        val[~slot_mask & (x[0] >= (mmi_length - self.device_length / 2)) & (np.abs(x[1]) < taper_height(x[0])) & converter_mask] = 1
        
        # Add in input/output wg's for convenience
        val[(x[0] <= (mmi_length - self.device_length / 2)) & (np.abs(x[1]) < self.strip_width/2)] = 1
        val[~slot_mask & (x[0] >= (self.device_length / 2)) & (np.abs(x[1]) < (2*self.slot_rail_width + self.slot_width)/2)] = 1
        
        return val
    
# %%% Simulation setup %%% 
    
class DengSymMMIStripToSlotSimulation(Simulation):
    def __init__(self,
                 max_device_length,
                 max_device_width,
                 strip_width=0.400,
                 slot_width=0.100,
                 slot_rail_width=0.150,
                 mat_height=0.250,
                 ridge_height=0.00,
                 center_wavelength=1.55,
                 wavelengths=[1.55],
                 gaussian_width=50,
                 optimize=False,
                 catch_errors=False,
                 render=False):
        Simulation.__init__(self, "Deng", working_dir="deng_wd", catch_errors=catch_errors)
        
        # Constraints that help setup the design region
        self.max_device_length = max_device_length
        self.max_device_width = max_device_width
        
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
        self.optimize = optimize
        self.catch_errors = catch_errors
        self.render = render
        
    def run(self, parameters):
        # Convergence parameters (included with the parameter list so automated convergence testing can change them around)
        straight_length = parameters["straight_length"]
        pml_x_thickness = parameters["pml_x_thickness"]
        pml_y_thickness = parameters["pml_y_thickness"]
        to_pml_x = parameters["to_pml_x"]
        to_pml_y = parameters["to_pml_y"]
        resolution = parameters["resolution"]
        min_run_time = parameters["min_run_time"]
        device_length = parameters["device_length"]
        min_run_time = parameters["min_run_time"]
        fields_decay_by = parameters["fields_decay_by"]
        
        # Geometry parameters
        mmi_length = parameters["mmi_length"]
        mmi_width = parameters["mmi_width"]
        
        # Reconstruct everything
        sx = 2 * (pml_x_thickness + to_pml_x) + self.max_device_length + 2 * straight_length
        sy = 2 * (to_pml_y + pml_y_thickness) + self.max_device_width
        cell = mp.Vector3(sx, sy)
        
        # Boundary conditions
        boundary_layers = [mp.PML(pml_x_thickness, direction=mp.X),
                           mp.PML(pml_y_thickness, direction=mp.Y)]
        
        # Our device is symmetric, mirror symmetry to speed up computation
        symmetries = []

        # Materials
        Si = mp.Medium(index=3.4)
        SiO2 = mp.Medium(index=1.44)
        
        # Geometry
        actual_L_straight = (sx - self.max_device_length) / 2

        # Sources and such
        parity = mp.ODD_Y + mp.EVEN_Z
        
        input_monitor_pt = mp.Vector3(-0.5 * (self.max_device_length + straight_length), 0)
        source_pt = mp.Vector3(-0.5 * self.max_device_length - 0.75 * straight_length, 0)
        output_monitor_pt = mp.Vector3(0.5 * (self.max_device_length + straight_length), 0)

        sources = [mp.EigenModeSource(src=mp.GaussianSource(1/self.center_wavelength, width=self.gaussian_width),
                                      center=source_pt,
                                      size=mp.Vector3(y=sy-2*pml_y_thickness),
                                      eig_match_freq=True,
                                      eig_parity=parity)]

        # Design region setup (using the pdt tools)
        meep_dr_nx = int(resolution * sx)
        meep_dr_ny = int(resolution * sy)
        
        converter = DengSymMMIStripToSlot(self.strip_width, self.slot_width, self.slot_rail_width, device_length, self.mat_height, self.ridge_height)
        design_region = DesignRegion([sx, sy], [meep_dr_nx, meep_dr_ny])
        
        # Design region setup (specific to MEEP)
        meep_design_variables = mp.MaterialGrid(mp.Vector3(meep_dr_nx,meep_dr_ny),SiO2,Si,grid_type='U_MEAN')
        meep_design_region = mpa.DesignRegion(meep_design_variables,volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy, 0)))

        # Material function
        material_function = lambda x: (1.44 + (3.4-1.44)*converter(np.asarray([[x[0]], [x[1]]]), params=dict(mmi_length=mmi_length, mmi_width=mmi_width)))**2
        dr_geometry=mp.Block(size=meep_design_region.size, epsilon_func=material_function)
        
        sim = mp.Simulation(resolution=resolution,
                                 cell_size=cell,
                                 boundary_layers=boundary_layers,
                                 geometry=[dr_geometry],
                                 sources=sources,
                                 eps_averaging=True, # See what this does
                                 default_material=SiO2,
                                 symmetries=symmetries)
        
        # Input and output monitors
        input_flux = sim.add_flux(1.0/self.center_wavelength, np.max(self.wavelengths) - np.min(self.wavelengths), len(self.wavelengths), mp.FluxRegion(center=input_monitor_pt,size=mp.Vector3(y=sy-2*pml_y_thickness)))
        output_flux = sim.add_flux(1.0/self.center_wavelength, np.max(self.wavelengths) - np.min(self.wavelengths), len(self.wavelengths), mp.FluxRegion(center=output_monitor_pt,size=mp.Vector3(y=sy-2*pml_y_thickness)))
        
        # Plot it
        plt.figure()
        sim.plot2D()
        
        if self.render:
            # Run and collect field data
            field_data_x = []
            field_data_y = []
            field_data_z = []
            def collect_fields(mp_sim):
                field_data_x.append(mp_sim.get_efield_x())
                field_data_y.append(mp_sim.get_efield_y())
                field_data_z.append(mp_sim.get_efield_z())
            
            sim.run(mp.at_every(resolution//4, collect_fields), 
                         until_after_sources=mp.stop_when_fields_decayed(min_run_time,mp.Ey,output_monitor_pt,fields_decay_by))
            
            # Render
            for comp in ["x", "y", "z"]:
                if comp == "x":
                    field_data = field_data_x
                elif comp == "y":
                    field_data = field_data_y
                elif comp == "z":
                    field_data = field_data_z
                
                render = Render("{working_dir}/fields_{parity}_{comp}.gif".format(working_dir=self.working_dir, parity=parity, comp=comp))
                max_field = np.max(np.abs(np.real(field_data)))
                for i, data in enumerate(field_data):
                    print("rendering {i}/{total}".format(i=i, total=len(field_data)))
                    fig = plt.figure(dpi=200)
                    plt.imshow(sim.get_epsilon().transpose(), interpolation='spline36', cmap='Greys')
                    plt.imshow(np.real(data).transpose(), vmin=-max_field, vmax=max_field, interpolation='spline36', cmap='RdBu', alpha=0.9)
                    render.add(fig)
                    plt.close()
                render.render(10)
        else:
            sim.run(until_after_sources=mp.stop_when_fields_decayed(min_run_time,mp.Ey,output_monitor_pt,fields_decay_by))
                                
        # Collect and process flux data (for the forward direction)
        input_coeffs = sim.get_eigenmode_coefficients(input_flux,[1],eig_parity=parity).alpha[0, :, 0]
        output_coeffs = sim.get_eigenmode_coefficients(output_flux,[1],eig_parity=parity).alpha[0, :, 0]
                    
        transmission = np.mean(np.abs(output_coeffs)**2 / np.abs(input_coeffs)**2)

        # Reset
        sim.reset_meep()
        
        return Result(parameters, transmission=transmission)
    
    def process(self, result, parameters):
        return result

if __name__ == "__main__":
    max_length = 12
    max_width = 5
    
    device_length = 4
    mmi_length = 1.7849179984327148
    mmi_width = 0.5332307718723385

    # Simulate our design    
    sim = DengSymMMIStripToSlotSimulation(max_length, max_width, render=False)
    
    # Parameters
    parameters = {
        "straight_length" : 3,
        "pml_x_thickness" : 1,
        "pml_y_thickness" : 1,
        "to_pml_x" : 1,
        "to_pml_y" : 1,
        "resolution" : 64,
        "min_run_time" : 300,
        "fields_decay_by" : 1e-9,
        "device_length" : device_length,
        "mmi_length" : mmi_length,
        "mmi_width" : mmi_width
    }
    
    '''
    result = sim.oneOff(parameters)
    print("transmission: {transmission}".format(transmission=result.values["transmission"]))
    '''
    
    strip2slot = DengSymMMIStripToSlot(sim.strip_width, sim.slot_width, sim.slot_rail_width, device_length, sim.mat_height, sim.ridge_height)
    design_region = DesignRegion([max_length, max_width], [int(parameters["resolution"] * max_length), int(parameters["resolution"] * max_width)])    
        
    optimizer = ScipyGradientOptimizer(sim, design_region, strip2slot, "transmission", ["mmi_length", "mmi_width"], strategy="maximize")
    optimizer.optimize(parameters)
    