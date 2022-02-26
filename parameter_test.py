#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:13:57 2022

@author: skenning
"""

import numpy as np
from pdt.core import *

class TestSimulation(Simulation):
    def __init__(self):
        Simulation.__init__(self, "test_simulation")
        
        # Do one-time initialization of simulation backend...
        
    def draw(self, parameters):
        x, y, z = parameters["x"], parameters["y"], parameters["z"]
        
        if z == 0:
            raise ValueError("z cannot be zero")
            
        # Do some 'drawing'...
    
    def run(self, parameters):
        x, y, z = parameters["x"], parameters["y"], parameters["z"]
        
        if x == 0:
            raise ValueError("x cannot be zero when running")
            
        return Result(parameters, f=x+y+z) # 'run' the simulation by doing x+y+z
    
    def process(self, result, parameters):     
        x, y, z = parameters["x"], parameters["y"], parameters["z"]
        
        if y == 7:
            raise ValueError("y cannot be 7")
            
        result.values["f"] = 0
        return result
    
    
if __name__ == "__main__":
    parameters = CSVParameterLoader("parameter_test.csv").parameters
    print("parameters:", parameters)
    
    sim = TestSimulation()
    sim.basicSweep(parameters)