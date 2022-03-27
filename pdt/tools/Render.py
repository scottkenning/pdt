#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 12:23:28 2022

@author: skenning
"""

import matplotlib.pyplot
import PIL
import matplotlib.pyplot as plt

class Render:
    def __init__(self, fname):
        self.fname = fname
        self.frames = []
        
    def add(self, plt_fig):
        plt_fig.canvas.draw()
        self.frames.append(PIL.Image.frombytes('RGB', plt_fig.canvas.get_width_height(),  plt_fig.canvas.tostring_rgb()))
        
    def render(self, duration):
        if len(self.frames) > 1:
            self.frames[0].save(self.fname, format="GIF", append_images=self.frames[1:], save_all=True, duration=duration, loop=0)
            
class ProgressRender:
    def __init__(self, fname, b_names, **fig_options):
        self.fname = fname
        self.b_names = b_names
        self.fig_options = fig_options
        
        self.frames = []
        
    def add(self, b, f0, jac):
        self.frames.append((b, f0, jac))
        
    def renderCurrent(self, fig=None, axs=None, upto=None):
        fig = None
        ax1, ax2, ax3 = None, None, None
        if axs:
            ax1, ax2, ax3 = axs
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, **self.fig_options)
        
        # Formatting and stuff
        ax1.set_title("parameters")
        ax2.set_title("figure of merit")
        ax3.set_title("gradient")
        
        ax3.set_xlabel("iteration")
        
        # Clip the frames to only render up to "upto"
        frames = None
        if upto is not None:
            frames = self.frames[:upto]
        else:
            frames = self.frames
        
        # Plot it
        for b_it, b_name in enumerate(self.b_names):
            b_i = []
            jac_i = []
            
            for b, f, jac in frames:
                b_i.append(b[b_it])
                jac_i.append(jac[b_it])
                
            ax1.plot(range(1, len(b_i)+1), b_i, label="{b_name}".format(b_name=b_name))
            ax3.plot(range(1, len(jac_i)+1), jac_i, label="{b_name}".format(b_name=b_name))
            
            ax1.legend()
            ax3.legend()
            
        f0 = []
        for _, f, _ in frames:
            f0.append(f)
        
        ax2.plot(range(1, len(frames)+1), f0)
        
        if fig:
            fig.savefig(self.fname)
            return fig
        
    def renderWithDesignEvolution(self, design_region, design, duration):
        render = Render(self.fname)
        
        for i in range(2, len(self.frames)):
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, **self.fig_options)
            
            # Draw the design
            b = dict()
            for b_name_it, b_name in enumerate(self.b_names):
                b[b_name] = self.frames[i][0][b_name_it]
                
            design_region.plotMaterialFunction(design, b, 0, ax=ax0)
            
            # Draw upto the current state
            self.renderCurrent(fig=None, axs=(ax1, ax2, ax3), upto=i+1)
            
            render.add(fig)
            plt.close(fig)
            
        render.render(duration)
        
if __name__ == "__main__":
    f0 = [1, 2, 3, 4]
    b = [[1, 4], [2, 3], [3, 2], [4, 1]]
    jac = [[1, 4], [2, 3], [3, 2], [4, 1]]
    
    pr = ProgressRender("test.png", ["one", "two"], figsize=(5, 10))
    for i in range(4):
        pr.add(b[i], f0[i], jac[i])
    pr.renderCurrent()
    