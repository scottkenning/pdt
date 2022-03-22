#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:47:42 2022

@author: skenning
"""

import numpy as np
import matplotlib.pyplot as plt

from pdt.opt import MaterialFunction, DesignRegion
from pdt.core import Simulation, ParameterChangelog

import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa

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
            "mmi_width" : (0, 1, 100),
            "mmi_length" : (0, 1, 100)
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
        # For MMI section
        val[(np.abs(x[1]) < mmi_width/2) & (x[0] < (mmi_length - self.device_length / 2))] = 1
        
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
        
        val[~slot_mask & (x[0] >= (mmi_length - self.device_length / 2)) & (np.abs(x[1]) < taper_height(x[0]))] = 1
        
        # Fantastic, it works well
        
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
                 optimize=False,
                 catch_errors=False):
        Simulation.__init__(self, "DengSymMMIStripToSlotSimulation", working_dir="dsmmis2s_working_dir", catch_errors=catch_errors)
        
        # Constraints that help setup the design region
        self.max_device_length = max_device_length
        self.max_device_width = max_device_width
        
        # Known parameters, based off of the platform
        self.strip_width = strip_width
        self.slot_width = slot_width
        self.slot_rail_width = slot_rail_width
        self.mat_height = mat_height
        self.ridge_height = ridge_height
        
        # Debug stuff
        self.optimize = optimize
        self.catch_errors = catch_errors
        
        # Globally held state
        self.parameter_changelog = ParameterChangelog()
        self.sim = None
        
    def run(self, parameters):
        # Convergence parameters (included with the parameter list so automated convergence testing can change them around)
        straight_length = parameters["straight_length"]
        pml_x_thickness = parameters["pml_x_thickness"]
        pml_y_thickness = parameters["pml_y_thickness"]
        to_pml_x = parameters["to_pml_x"]
        to_pml_y = parameters["to_pml_y"]
        resolution = parameters["resolution"]
        min_run_time = parameters["min_run_time"]
        
        # Geometry parameters
        mmi_length = parameters["mmi_length"]
        mmi_width = parameters["mmi_width"]
        
        # Does the simulation need to be updated?
        if self.parameter_changelog.changesExclude(["mmi_length", "mmi_width"]):
            self._log_info("Convergence parameters changed, rebuilding simulation")
            
            # Wipe all MEEP state
            if self.sim:
                self.sim.reset_meep()
                
                self.sim = None
                
            # Reconstruct everything
            sx = 2 * (pml_x_thickness + to_pml_x) + self.max_device_length + 2 * straight_length
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
            actual_L_straight = (sx - self.max_device_length) / 2

            small_wg = mp.Block(mp.Vector3(actual_L_straight, self.taper_w1), center=mp.Vector3(-self.max_device_length/2 - actual_L_straight/2, 0), material=Si)
            large_wg = mp.Block(mp.Vector3(actual_L_straight, self.taper_w2), center=mp.Vector3(self.max_device_length/2 + actual_L_straight/2, 0), material=Si)
            
            # Sources and such
            input_monitor_pt = mp.Vector3(-0.5 * (self.max_device_length + straight_length), 0)
            source_pt = mp.Vector3(-0.5 * self.max_device_length - 0.75 *  + straight_length, 0)
            output_monitor_pt = mp.Vector3(0.5 * (self.max_device_length + straight_length), 0)

            sources = [mp.EigenModeSource(src=mp.GaussianSource(1/self.center_wavelength, width=self.gaussian_width),
                                          center=source_pt,
                                          size=mp.Vector3(y=sy-2*pml_y_thickness),
                                          eig_match_freq=True,
                                          eig_parity=mp.ODD_Z+mp.EVEN_Y)]

if __name__ == "__main__":
    length = 6
    max_width = 5
    
    mmi_length = 1.38
    mmi_width = 1.24
    
    dr = DesignRegion((length, max_width), (1000, 1000))
    
    strip2slot = DengSymMMIStripToSlot(strip_width=0.400, 
                                       slot_width=0.100,
                                       slot_rail_width=0.150, 
                                       device_length=length, 
                                       mat_height=0.250, 
                                       ridge_height=0.00)
    
    dr.plotMaterialFunction(strip2slot, dict(mmi_length=mmi_length, mmi_width=mmi_width))
