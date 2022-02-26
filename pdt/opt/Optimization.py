#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:35:14 2022

@author: skenning
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

class MaterialFunction:
    def __init__(self, dx_hints: dict[str, (float, float, int)], how_hints=np.linspace):
        self.dx_hints = dx_hints
        self.how_hints = how_hints
        # dx_hints has tuples (min, max, number of points) 
    
    def evalModel(self, x, **kwargs):
        pass
    
    def _get_hint_grid(self, name):
        return self.how_hints(self.dx_hints[name][0], self.dx_hints[name][1], self.dx_hints[name][2])
    
    def __call__(self, x, **kwargs):
        result = self.evalModel(x, **kwargs)
        if result is None:
            return ValueError("MaterialFunction.evalModel returned None. Are you sure this function has been overriden?")
        else:
            return result

class DesignRegion:
    def __init__(self, size, N):
        if len(N) != len(size):
            raise ValueError("The DesignRegion's size ({size}) must have the same length as N ({N})".format(size=size, N=N))
        
        self.dim = len(size)        
        self.size = size
        self.N = N
        
        self.dx = np.asarray([size[i] / N[i] for i in range(self.dim)])
        
        self.n_grid = np.meshgrid(*[np.arange(0, self.N[d]) for d in range(self.dim)])
        for i, axis in enumerate(self.n_grid):
            self.n_grid[i] = axis.flatten()
            
        self.x_grid = self.mapGridToReal(self.n_grid)
    
    def _check_coord(self, c):
        c = np.asarray(c)
        
        if self.dim == 1 and len(c.shape) == 1:
            c = np.asarray([c])
        
        if len(c.shape) != 2:
            raise ValueError("DesignRegion.map functions requires a parameter c with len(c.shape) equal to 2")
        if c.shape[0] != self.dim:
            raise ValueError("DesignRegion.map functions requires a parameter c with c.shape[0] equal to the dimensionality of the region (c.shape[0] == {shape})".format(shape=c.shape[0]))
            
        return c
    
    def mapRealToGrid(self, x):
        x = self._check_coord(x)
        n = np.zeros(x.shape)
        
        for d in range(self.dim):
            n[d,:] = np.clip(np.round((x[d,:] + self.size[d] / 2 - self.dx[d] / 2) * self.N[d] / self.size[d]), 0, self.N[d] - 1)
            
        return n
            
    def mapGridToReal(self, n):
        n = self._check_coord(n)
        x = np.zeros(n.shape, dtype=np.float128)
        
        for d in range(self.dim):
            x[d,:] = np.clip(self.size[d] * n[d,:] / self.N[d] + self.dx[d] / 2 - self.size[d]/2, -self.size[d] / 2, self.size[d] / 2)
            
        return x
    
    def evalMaterialFunction(self, mat_func, **kwargs):
        return mat_func(self.x_grid, **kwargs).reshape(self.N) 
    
    def evalMaterialFunctionDerivative(self, mat_func, x: dict[str, float], dx: dict[str, float]):
        # Ok, so here is where things start to get a little complex. We need to
        # find du/db, where u is an individual grid point's change when one of
        # the parameters to the material function is perturbed. 
        order = x.keys()
        current = self.evalMaterialFunction(mat_func, **x)
                
        if len(x) != len(dx):
            raise ValueError("DesignRegion.evalMaterialFunctionDerivative must be have len(x) == len(dx)")
        
        # du/db_i (du is an array the size of the design region)
        du_db = []
        for i, current_name in enumerate(order):
            # We want the perturbation algorithm to be smart, that means if it generates 
            # a device that is equivalent with respect to the underlying discrete material grid,
            # it will up the perturbation based off of the material hints.
            
            possible_dx_i_s = mat_func._get_hint_grid(current_name)
            for j, possible_dx_i in enumerate(possible_dx_i_s):
                # Perturb
                perturbed_x = copy.deepcopy(x)
                perturbed_x[current_name] += possible_dx_i
                
                # Evaluate
                u_b_i = self.evalMaterialFunction(mat_func, **perturbed_x)
                
                # Check to see if it is sufficiently different to be useful
                difference = np.sum(np.abs(current - u_b_i))
                if not np.isclose(0, difference):
                    # sufficiently different
                    du_db.append((u_b_i - current) / possible_dx_i)
                    break
                else:
                    if j == len(possible_dx_i_s) - 1:
                        # The hint range does not perturb the design near x
                        # Assume the derivative is zero
                        du_db.append(np.zeros(self.N))
        
        # Now that we have that array
                        
                
                
                
            
            du_db_i = self.evalMaterialFunction
            
if __name__ == "__main__":
    dr = DesignRegion([100, 100], [100, 100])
        
    x = np.linspace(0, 100, 100) - 50 + 0.5
    y = x
    
    x, y = np.meshgrid(x, y)
    coord = np.asarray([x.flatten(), y.flatten()])
    print(coord.shape)
    
    n = dr.mapRealToGrid(coord)
    
    returned_coord = dr.mapGridToReal(n)
    
    print(np.max(coord[1] - returned_coord[1]))
    
    