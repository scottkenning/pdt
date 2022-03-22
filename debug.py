#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:12:50 2022

@author: skenning
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

from pdt.tools.Render import Render
from pdt.opt.Optimization import DesignRegion, LegendreTaperMaterialFunction, MaterialFunction

def plot_transmission_brute():
    with h5.File("lts_working_dir/LegendreTaperSimulation_brute.hdf5", 'r') as f:
        transmissions = np.asarray([np.mean(f[sim]["transmission"]) for sim in f.keys()])
        coeffs = np.asarray([f[sim].attrs["polynomial_coeffs"][0] for sim in f.keys()])
        
        order = coeffs.argsort()
        coeffs = coeffs[order]
        transmissions = transmissions[order]
        
        plt.figure()
        plt.scatter(coeffs, transmissions)
        plt.savefig("transmission.png")
        
        plt.figure()
        plt.scatter(coeffs[:-1], np.diff(transmissions)/np.diff(coeffs))
        plt.savefig("dtransmission.png")
        
def plot_transmission_opt():
    with h5.File("lts_working_dir/LegendreTaperSimulation_opt.hdf5", 'r') as f:
        transmissions = np.asarray([np.mean(f[sim]["f0"]) for sim in f.keys()])
        grads = np.asarray([np.mean(f[sim]["dJ_db"]) for sim in f.keys()])
        coeffs = np.asarray([f[sim].attrs["polynomial_coeffs"][0] for sim in f.keys()])
        
        order = coeffs.argsort()
        coeffs = coeffs[order]
        grads = grads[order]
        transmissions = transmissions[order]
        
        plt.figure()
        plt.scatter(coeffs, transmissions)
        plt.savefig("transmission.png")
        
        plt.figure()
        diff = np.diff(transmissions)/np.diff(coeffs)
        plt.scatter(coeffs[:-1], diff)
        plt.savefig("dtransmission.png")
        
        plt.figure()
        plt.semilogy(coeffs[:-1], (3.4**2 - 1.44**2)*((64 * 2*64)) * np.abs(grads[:-1]/diff))
        plt.savefig("adjointgrad.png")
        
        print(grads)      

def plot_data(fname):
    with h5.File(fname, mode='r') as f:
        for sim in f.keys():           
            coeff = f[sim].attrs["polynomial_coeffs"][0]
            sigma = f[sim].attrs["sigma"]
            all_du_db = np.squeeze(f[sim]["all_du_db"])
            dJ_du = np.squeeze(f[sim]["dJ_du"])
            dJ_db_adjoint = np.squeeze(f[sim]["dJ_db"])
            resolution = f[sim].attrs["resolution"]
            
            taper_length = 2
            taper_w2 = 1
            taper_w1 = 0.4
            
            # Design setup
            meep_dr_nx = int(resolution * taper_length)
            meep_dr_ny = int(resolution * taper_w2)
            
            taper = LegendreTaperMaterialFunction(1, [taper_length, taper_w2], taper_w1, taper_w2)
            design_region = DesignRegion([taper_length, taper_w2], [meep_dr_nx, meep_dr_ny])
            
            '''
            # Render an image of the geometry's perturbation
            render = Render("lts_working_dir/renders/du_db_{sim}.gif".format(sim=sim))
            for i in range(all_du_db.shape[0]):
                fig = plt.figure()
                plt.title("{coeff}:{i}".format(coeff=round(coeff, 2), i=i))
                plt.imshow(all_du_db[i, :, :].transpose())
                render.add(fig)
            render.render(30)
            '''
            
            plt.figure()
            plt.imshow(dJ_du)
            
            order, du_db, min_db, all_du_db = design_region.evalMaterialFunctionDerivative(taper, MaterialFunction.arrayToParams('b', [coeff]), sigma)
            all_du_db = np.squeeze(all_du_db)
            
            render = Render("lts_working_dir/renders/dJ_db_i_{sim}.gif".format(sim=sim))
            for i in range(all_du_db.shape[0]):
                dJ_db_i = np.squeeze(np.asarray([dJ_du * all_du_db[i,:,:]]))
                
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
                ax1.set_title("{coeff}:{i}:{dJ_db_i}".format(coeff=round(coeff, 2), i=i, dJ_db_i=round(np.sum(dJ_db_i), 10)))
                ax1.imshow(dJ_db_i)
                ax2.imshow(dJ_du)
                ax3.imshow(dJ_du > 0)
                ax4.imshow(all_du_db[i,:,:])
                render.add(fig)
                plt.close()
                
            render.render(30)
            
if __name__ == "__main__":
    plot_data("lts_working_dir/LegendreTaperSimulation.hdf5")