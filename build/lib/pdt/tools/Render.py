#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 12:23:28 2022

@author: skenning
"""

import matplotlib.pyplot
import PIL

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
        