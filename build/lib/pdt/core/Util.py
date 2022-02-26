#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 19:44:03 2022

@author: skenning
"""

import typing
import os
import pathlib
from mpi4py import MPI

def assert_parameter_lengths(parameters: dict[str, list[typing.Any]]):
    prev_length = None
    for key, item_list in parameters.items():
        if prev_length is None:
            prev_length = len(item_list)
        else:
            if prev_length != len(item_list):
                return False
    return True

def get_parameter_lengths(parameters: dict[str, list[typing.Any]]):
    if assert_parameter_lengths(parameters):
        if len(parameters):
            return len(parameters[list(parameters.keys())[0]])
        else:
            return 0
    else:
        raise ValueError("Parameter lengths are not the same")
        
def get_parameter_iteration(iteration, parameters: dict[str, list[typing.Any]]):
    if iteration < get_parameter_lengths(parameters):
        iteration_parameters = dict()
        
        for key, items in parameters.items():
            iteration_parameters[key] = items[iteration]
        return iteration_parameters
    else:
        raise ValueError("Parameter iteration is out of range")
        
def hash_parameter_iteration(parameter_iteration: dict[str, typing.Any]):
    return hash(frozenset(parameter_iteration.items()))

def make_path_exist(path: str):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

# Decorator that only allows a function to be called if it is the master process     
def is_master():
    return MPI.COMM_WORLD.Get_rank() == 0
 
def if_master(func):
    def inner_do(*args, **kwargs):
        return func(*args, **kwargs)
    def inner_dont(*args, **kwargs):
        pass
    
    if is_master():
        return inner_do
    else:
        return inner_dont
    

            