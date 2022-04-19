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
from scipy.ndimage import gaussian_filter
import scipy.optimize as sciopt
from pdt.core.Simulation import Simulation
import pdt.core.Util as Util
from pdt.tools.Render import Render, ProgressRender

class MaterialFunction:
    def __init__(self, db_hints: dict[str, (float, float, int)], how_hints=np.linspace):
        self.db_hints = db_hints
        self.how_hints = how_hints
        # dx_hints has tuples (min, max, number of points) 
    
    def evalModel(self, x, params: dict[str, float]):
        pass
    
    #def evalModelFlat(self, x, params: dict[str, float]):
    
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
    
    def paramListHelper(count: int, base: str):
        l = []
        for i in range(count):
            l.append("{b}{i}".format(b=base, i=i))
        return l

class DesignRegion:
    def __init__(self, size, N):
        if len(N) != len(size):
            raise ValueError("The DesignRegion's size ({size}) must have the same length as N ({N})".format(size=size, N=N))
        
        self.dim = len(size)        
        self.size = size
        self.N = N
        
        self.dx = np.asarray([size[i] / N[i] for i in range(self.dim)])
        self.dA = np.prod(self.dx)
                
        self.n_grid = np.meshgrid(*[np.arange(0, self.N[d]) for d in range(self.dim)])        
        self.x_grid = self.mapGridToReal(self.n_grid)
    
    def _check_coord(self, c):       
        if len(c) != self.dim:
            raise ValueError("DesignRegion.map functions requires a parameter c with len(c) equal to the dimensionality of the region (c.shape[0] == {shape})".format(shape=c.shape[0]))
          
    def scaleAdjointGradient(self, dJ_du, delta_epsilon):
        print(delta_epsilon / self.dA)
        return dJ_du * delta_epsilon / self.dA
          
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
    
    def evalMaterialFunction(self, mat_func, x: dict[str, float], sigma):
        return gaussian_filter(mat_func(self.x_grid, x), sigma)
    
    def plotMaterialFunction(self, mat_func, x: dict[str, float], sigma, ax=None):
        if self.dim != 2:
            raise ValueError("Cannot plotMaterialFunction unless dimensionality of the design region is 2")
        
        if ax:
            ax.set_title("material function")
            ax.imshow(self.evalMaterialFunction(mat_func, x, sigma), extent=(-self.size[0]/2, self.size[0]/2, -self.size[1]/2, self.size[1]/2))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            plt.figure()
            plt.title("material function")
            plt.imshow(self.evalMaterialFunction(mat_func, x, sigma), extent=(-self.size[0]/2, self.size[0]/2, -self.size[1]/2, self.size[1]/2))
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
        
    def plotMaterialFunctionDerivative(self, mat_func,  x: dict[str, float], sigma):
        order, du_db, _, _ = self.evalMaterialFunctionDerivative(mat_func, x, sigma)
        
        for item, du_db_i in zip(order, du_db):
            plt.figure()
            plt.title(r"$du/db$ for {item}".format(item=item))
            plt.imshow(du_db_i, extent=(-self.size[0]/2, self.size[0]/2, -self.size[1]/2, self.size[1]/2))
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
    
    def evalMaterialFunctionDerivative(self, mat_func, x: dict[str, float], sigma):
        # Ok, so here is where things start to get a little complex. We need to
        # find du/db, where u is an individual grid point's change when one of
        # the parameters to the material function is perturbed. 
        order = x.keys()
        current = self.evalMaterialFunction(mat_func, x, sigma)
        
        # du/db_i (du is an array the size of the design region)
        du_db = []
        min_db = []
        all_du_db = []
        for i, current_name in enumerate(order):
            # We want the perturbation algorithm to be smart, that means if it generates 
            # a device that is equivalent with respect to the underlying discrete material grid,
            # it will up the perturbation based off of the material hints.
            
            possible_dx_i_s = mat_func._get_hint_grid(current_name)
            all_du_db_i = []
            found = False
            for j, possible_dx_i in enumerate(possible_dx_i_s):
                # Perturb
                perturbed_x_positive = copy.deepcopy(x)
                perturbed_x_positive[current_name] += possible_dx_i
                
                perturbed_x_negative = copy.deepcopy(x)
                perturbed_x_negative[current_name] -= possible_dx_i
                
                '''
                # Evaluate
                u_b_i_positive = self.evalMaterialFunction(mat_func, perturbed_x_positive, sigma)
                u_b_i_negative = self.evalMaterialFunction(mat_func, perturbed_x_negative, sigma)
                all_du_db_i.append((u_b_i_positive - u_b_i_negative) / 2*possible_dx_i)
                
                # Check to see if it is sufficiently different to be useful
                if not found:
                    difference = np.sum(np.abs(u_b_i_positive - u_b_i_negative))
                    if not np.isclose(0, difference):
                        #print("perturbation for {current_name}: {possible_dx_i}".format(current_name=current_name, possible_dx_i=possible_dx_i))
                        # sufficiently different
                        du_db.append((u_b_i_positive - u_b_i_negative) / 2*possible_dx_i)
                        min_db.append(possible_dx_i)
                        found = True
                    else:
                        if j == len(possible_dx_i_s) - 1:
                            # The hint range does not perturb the design near x
                            # Assume the derivative is zero
                            du_db.append(np.zeros(self.N))
                            min_db.append(0)
                '''
                # Evaluate
                u_b_i_positive = self.evalMaterialFunction(mat_func, perturbed_x_positive, sigma)
                u_b_i = self.evalMaterialFunction(mat_func, x, sigma)
                all_du_db_i.append((u_b_i_positive - u_b_i) / possible_dx_i)
                
                # Check to see if it is sufficiently different to be useful
                if not found:
                    difference = np.sum(np.abs((u_b_i_positive - u_b_i)))
                    if not np.isclose(0, difference):
                        #print("perturbation for {current_name}: {possible_dx_i}".format(current_name=current_name, possible_dx_i=possible_dx_i))
                        # sufficiently different
                        du_db.append((u_b_i_positive - u_b_i) / possible_dx_i)
                        min_db.append(possible_dx_i)
                        found = True
                    else:
                        if j == len(possible_dx_i_s) - 1:
                            # The hint range does not perturb the design near x
                            # Assume the derivative is zero
                            du_db.append(np.zeros(self.N))
                            min_db.append(0)
                
            all_du_db.append(all_du_db_i)
        # Now that we have that array, we return it as our result
        return order, du_db, min_db, np.asarray(all_du_db)

class ScipyGradientOptimizer:
    def __init__(self, sim: Simulation, design_region, design, fom: str, jac: str, opt_parameters: list[str], method="L-BFGS-B", strategy="minimize", include_jac_key="include_jac"):
        self.sim = sim
        self.design_region = design_region # Function that returns the currently active design region!
        self.design = design # Function that returns the currently active design!
        self.fom = fom
        self.jac = jac
        self.opt_parameters = opt_parameters
        self.method = method
        self.strategy = strategy
        self.include_jac_key = include_jac_key
        
        self.prev_run_results = dict()
      
    def _scipy_to_pdt(self, start_parameters, b):
        # Convert from scipy parameter format to pdt format
        params = copy.deepcopy(start_parameters)
        for i, opt_parameter in enumerate(self.opt_parameters):
            params[opt_parameter] = b[i]
        return params
    
    def _apply_strategy(self, val):
        val = np.asarray(val)
        if self.strategy == "minimize":
            return val
        elif self.strategy == "maximize":
            return -val
        else:
            raise ValueError("Unknown optimization strategy of {strategy}".format(strategy=self.strategy))
            
    def _strip_objective_rvalue(rvalue, include_jac):
        if include_jac:
            return rvalue
        else:
            if isinstance(rvalue, tuple):
                return rvalue[0]
            else:
                return rvalue
    
    def _make_render(self, progress_render_fname, progress_render_fig_kwargs):
        if progress_render_fname:
            return ProgressRender("{working_dir}/{fname}".format(working_dir=self.sim.working_dir, fname=progress_render_fname), self.opt_parameters, **progress_render_fig_kwargs)
        else:
            return None
    
    def _render(self, progress_render, b, f0, jac, progress_render_fancy, progress_render_duration):
        if progress_render:
            progress_render.add(b, f0, jac)
            
            if progress_render_fancy:
                progress_render.renderWithDesignEvolution((self.design_region)(), (self.design)(), progress_render_duration)
            else:
                progress_render.renderCurrent()
    
    def optimize(self, start_parameters: dict[str, float], finite_difference, progress_render_fname=None, progress_render_fig_kwargs=dict(), progress_render_fancy=True, progress_render_duration=10, **kwargs):
        start_b = list(self._get_opt_parameters(start_parameters).values())
        
        self.sim._log_info("optimizer starting at {start_b}".format(start_b=start_b))
        progress_render = self._make_render(progress_render_fname, progress_render_fig_kwargs)

        # Scipy compatible objective and jacobian functions, with logging included
        def _objective(b):
            params = self._scipy_to_pdt(start_parameters, b)
            f0 = ScipyGradientOptimizer._strip_objective_rvalue(self.objective(params, not finite_difference), False)
            return self._apply_strategy(f0)
        
        def _jacobian(b):
            params = self._scipy_to_pdt(start_parameters, b)
            f0, jac = self.jacobian(params, finite_difference)
            self._render(progress_render, b, f0, jac, progress_render_fancy, progress_render_duration)
            return self._apply_strategy(jac)
        
        # Now we call the scipy optimization routine
        sciopt.minimize(_objective, start_b, jac=_jacobian, **kwargs)
            
    def objective(self, params, include_jac):
        if Util.hash_parameter_iteration(params) in self.prev_run_results.keys():
            if include_jac:
                prev_result = self.prev_run_results[Util.hash_parameter_iteration(params)]
                
                if isinstance(prev_result, tuple):
                    return ScipyGradientOptimizer._strip_objective_rvalue(prev_result, True)
                else:
                    pass # The function will not return and will go and perform a new run
                        
            else:
                prev_result = self.prev_run_results[Util.hash_parameter_iteration(params)]
                return ScipyGradientOptimizer._strip_objective_rvalue(prev_result, False)
                
        # New run
        new_run_params = copy.deepcopy(params)
        new_run_params[self.include_jac_key] = include_jac
        result = self.sim.oneOff(new_run_params)
        
        if include_jac:            
            self.prev_run_results[Util.hash_parameter_iteration(params)] = (result.values[self.fom], result.values[self.jac])
            f0 = np.asarray(result.values[self.fom])
            jac = np.asarray(result.values[self.jac])
            self.sim._log_info("optimizer evaluated {b}: f0={f0}".format(b=self._get_opt_parameters(params), f0=f0))
            
            return f0, jac
        else:
            self.prev_run_results[Util.hash_parameter_iteration(params)] = result.values[self.fom]            
            f0 = np.asarray(result.values[self.fom])            
            self.sim._log_info("optimizer evaluated {b}: f0={f0}".format(b=self._get_opt_parameters(params), f0=f0))
            
            return f0
        

    def _get_opt_parameters(self, params):
        b = dict()
        for opt_parameter in self.opt_parameters:
            b[opt_parameter] = params[opt_parameter]  
        return b
    
    def _perturb_opt_parameters(self, params, order, min_db):
        perturbations = []
        
        for i, parameter in enumerate(order):
            perturbation = copy.deepcopy(params)
            perturbation[parameter] += min_db[i]
            
            perturbations.append(perturbation)
            
        return perturbations
    
    def jacobian(self, params, finite_difference):
        if finite_difference:
            f0 = self.objective(params, False)
            
            # Get the minimum db
            b = self._get_opt_parameters(params)        
            order, _, min_db, _ = (self.design_region)().evalMaterialFunctionDerivative((self.design)(), b, 0)
            
            # Perturb the optimization parameters
            delta_params = self._perturb_opt_parameters(params, order, min_db)
            
            # Compute the jacobian
            df_db = []
            for i, parameter in enumerate(order):
                b_i = self._get_opt_parameters(delta_params[i])
                db_i = min_db[i]
                f0_i = self.objective(delta_params[i], False)
                
                df_db.append((f0_i - f0) / db_i)
            df_db = np.asarray(df_db)
            
            self.sim._log_info("optimizer evaluated {b}: jac={jac}".format(b=b, jac=df_db))
            return f0, df_db
        else: 
            f0, jac = self.objective(params, True)
            return f0, jac
            
# A nice demonstration example. Taper boundaries are defined by the odd Legendre Polynomials
from scipy.special import legendre
class LegendreTaperMaterialFunction(MaterialFunction):
    def __init__(self, order, dim, w1, w2):
        MaterialFunction.__init__(self, db_hints=MaterialFunction.hintHelper(count=order, base='b', prototype=(.25, 1, 100)))
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
            
        return val
    
    
