#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 11:13:57 2022

@author: skenning
"""

import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

setup(name='pdt',
      version='0.0',
      description='Photonic design tools to automate common tedious tasks.',
      author='Scott Kenning',
      author_email='skenning@purdue.edu',
      install_requires=[],
      packages=['pdt.core', 'pdt.opt', 'pdt.tests', 'pdt.docs', 'pdt.backend'],
      zip_safe=False)