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
        
    def draw(self, parameters):
        pass
    
    def run(self, parameters):
        pass
    
    def process(self, result, parameters):
        pass
    
    
if __name__ == "__main__":
    pass