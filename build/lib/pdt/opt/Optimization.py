#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:35:14 2022

@author: skenning
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.pyplot as plt

class MaterialFunction:
    def __init__(self, db_hints: dict[str, (float, float, int)], how_hints=np.linspace):
        self.db_hints = db_hints
        self.how_hints = how_hints
        # dx_hints has tuples (min, max, number of points) 
    
    def evalModel(self, x, params: dict[str, float]):
        pass
    
    def _get_hint_grid(self, name):
        return self.how_hints(self.db_hints[name][0], self.db_hints[name][1], self.db_hints[name][2])
    
    def __call__(self, x, params: dict[str, float]):
        result = self.evalModel(x, params)
        if result is None:
            return ValueError("MaterialFunction.evalModel returned None. Are you sure this function has been overriden?")
        else:
            return result
        
    def hintHelper(count: int, base: str, prototype: (float, float, int)):
        hints = dict()
        for i in range(count):
            hints["{b}{i}".format(b=base, i=i)] = prototype
        return hints
    
    def paramsToArray(count: int, base: str, params):
        arr = []
        for i in range(count):
            arr.append(params["{b}{i}".format(b=base, i=i)])
        return arr
    
    def arrayToParams(base: str, arr):
        params = dict()
        for i, a in enumerate(arr):
            params["{b}{i}".format(b=base, i=i)] = a
        return params

class DesignRegion:
    def __init__(self, size, N):
        if len(N) != len(size):
            raise ValueError("The DesignRegion's size ({size}) must have the same length as N ({N})".format(size=size, N=N))
        
        self.dim = len(size)        
        self.size = size
        self.N = N
        
        self.dx = np.asarray([size[i] / N[i] for i in range(self.dim)])
                
        self.n_grid = np.meshgrid(*[np.arange(0, self.N[d]) for d in range(self.dim)])        
        self.x_grid = self.mapGridToReal(self.n_grid)
    
    def _check_coord(self, c):       
        if len(c) != self.dim:
            raise ValueError("DesignRegion.map functions requires a parameter c with len(c) equal to the dimensionality of the region (c.shape[0] == {shape})".format(shape=c.shape[0]))
            
    def mapRealToGrid(self, x):
        self._check_coord(x)
        n = [np.zeros(x_it.shape, dtype=np.float128) for x_it in x]
        
        for d in range(self.dim):
            n[d] = np.clip(np.round((x[d] + self.size[d] / 2 - self.dx[d] / 2) * self.N[d] / self.size[d]), 0, self.N[d] - 1)
            
        return n
            
    def mapGridToReal(self, n):
        self._check_coord(n)
        x = [np.zeros(n_it.shape, dtype=np.float128) for n_it in n]
        
        for d in range(self.dim):
            x[d] = np.clip(self.size[d] * n[d] / self.N[d] + self.dx[d] / 2 - self.size[d]/2, -self.size[d] / 2, self.size[d] / 2)
            
        return x
    
    def evalMaterialFunction(self, mat_func, x: dict[str, float]):
        return mat_func(self.x_grid, x)
    
    def plotMaterialFunction(self, mat_func, x: dict[str, float]):
        if self.dim != 2:
            raise ValueError("Cannot plotMaterialFunction unless dimensionality of the design region is 2")
            
        plt.figure()
        plt.title("material function")
        plt.imshow(self.evalMaterialFunction(mat_func, x), extent=(-self.size[0]/2, self.size[0]/2, -self.size[1]/2, self.size[1]/2))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        
    def plotMaterialFunctionDerivative(self, mat_func,  x: dict[str, float]):
        order, du_db = self.evalMaterialFunctionDerivative(mat_func, x)
        
        for item, du_db_i in zip(order, du_db):
            plt.figure()
            plt.title(r"$du/db$ for {item}".format(item=item))
            plt.imshow(du_db_i, extent=(-self.size[0]/2, self.size[0]/2, -self.size[1]/2, self.size[1]/2))
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
    
    def evalMaterialFunctionDerivative(self, mat_func, x: dict[str, float]):
        # Ok, so here is where things start to get a little complex. We need to
        # find du/db, where u is an individual grid point's change when one of
        # the parameters to the material function is perturbed. 
        order = x.keys()
        current = self.evalMaterialFunction(mat_func, x)
        
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
                u_b_i = self.evalMaterialFunction(mat_func, perturbed_x)
                
                # Check to see if it is sufficiently different to be useful
                difference = np.sum(np.abs(current - u_b_i))
                if not np.isclose(0, difference):
                    #print("perturbation for {current_name}: {possible_dx_i}".format(current_name=current_name, possible_dx_i=possible_dx_i))
                    # sufficiently different
                    du_db.append((u_b_i - current) / possible_dx_i)
                    break
                else:
                    if j == len(possible_dx_i_s) - 1:
                        # The hint range does not perturb the design near x
                        # Assume the derivative is zero
                        du_db.append(np.zeros(self.N))
        
        # Now that we have that array, we return it as our result
        return order, du_db

from scipy.special import legendre
class LegendreTaperMaterialFunction(MaterialFunction):
    def __init__(self, order, dim, w1, w2):
        MaterialFunction.__init__(self, db_hints=MaterialFunction.hintHelper(count=order, base='b', prototype=(0, 1, 100)))
        self.order = order
        self.dim = dim    
        
        self.w1 = w1
        self.w2 = w2
        self.w_travel = (w2 - w1) / 2
        
        self.basis = [legendre(2*n + 1) for n in range(order)] 
        
    def evalModel(self, x, params: dict[str, float]):           
        s = 2 * x[0] / self.dim[0]
        w = self._eval_weighted(MaterialFunction.paramsToArray(self.order, 'b', params), np.asarray(s, dtype=np.float64))
        
        val = np.ones(x[0].shape)
        val[w + self.w1/2 < np.abs(x[1])] *= 0
        
        return val

    # Evaluation of the basis functions
    def _eval_weighted(self, beta, s):
        val = np.zeros(s.shape)
        def eval_basis(i):
            return self.w_travel * (self.basis[i](s) + 1) / 2

        # Evaluate for the degrees of freedom
        for i in range(self.order):
            val += beta[i] * eval_basis(i)
            
        # We mutiply everything by a scaling factor to assure that the basis values
        # sum to 1
        val /= np.sum(beta)
            
        return val

def test_all():
    dr = DesignRegion([10, 5], [1000, 500])
    ltmf = LegendreTaperMaterialFunction(3, (10, 5), 1, 5)
    
    params = {'b0': 1,
              'b1': 0,
              'b2': 0
             }
    
    dr.plotMaterialFunction(ltmf, params)
    dr.plotMaterialFunctionDerivative(ltmf, params)
    
    params = {'b0': 1,
              'b1': 0.1,
              'b2': 0
             }
    dr.plotMaterialFunction(ltmf, params)
    params = {'b0': 1,
              'b1': 0,
              'b2': 0.1
             }
    dr.plotMaterialFunction(ltmf, params)

    
    

if __name__ == "__main__":
    test_all()
    