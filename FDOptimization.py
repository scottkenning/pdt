#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:14:25 2022

@author: skenning
"""

class FDOptimization:
    def __init__(self, dx, bounds):
        self.dx = dx
    
    def gradient(self, x):
        