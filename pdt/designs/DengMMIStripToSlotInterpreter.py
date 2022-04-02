#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:41:10 2022

@author: skenning
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

def plotConvergence(fname="WD_DengMMIStripToSlot/DengMMIStripToSlot_Convergence.hdf5"):
    with h5.File(fname) as f:
        # Collect the information from the simulation file
        f0s = []
        grads = []
        grad_names = []
        resolutions = []
        for sim in f:
            sim = f[sim]
            f0s.append(np.squeeze(sim["f0"]))
            grads.append(np.squeeze(sim["dJ_db"]))
            resolutions.append(np.squeeze(sim.attrs["resolution"]))
            
            if not len(grad_names):
                grad_names = sim.attrs["opt_parameters"]
        f0s = np.asarray(f0s)
        grads = np.asarray(grads)
        
        # Plot it appropriately
        plt.figure()
        plt.title("f0")
        plt.scatter(resolutions, f0s)
            
        plt.figure(figsize=(10, 10))
        plt.title("grad")
        plt.yscale("log")
        for i in range(grads.shape[1]):
            if grad_names[i] in ["mmi_length", "mmi_width", "input_ridge_runup"]:
                plt.scatter(resolutions, np.abs(grads[:,i]), label=grad_names[i])
        plt.legend()
            
if __name__ == "__main__":
    plotConvergence()