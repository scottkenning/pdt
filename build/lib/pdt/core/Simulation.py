#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:47:39 2022

@author: skenning
"""

import typing
import pandas as pd
import logging
import h5py
import time
import numpy as np

from pdt.core.Util import *
from pdt.core.Util import if_master

'''
The Result class only 'does' things when it is in the master process.
Otherwise, it would store redundant information and just take up memory.
'''
class Result:
    def __init__(self, parameters: dict[str, typing.Any], **values):
        if is_master():
            self.parameters = parameters
            self.values = values
    
    def _get_unique_name(timestamp: float, parameters: dict[str, typing.Any]):
        # Generates a hash based on the timestamp and parameters
        return str(hash(frozenset([timestamp, frozenset(parameters.items())])))
    
    @if_master
    def _save_data(group, name, value, logger):
        # If the data is not in a numpy array, we (try to) convert it to one
        if not isinstance(value, np.ndarray):
            try:
                value = np.asarray(value)
            except Exception as e:
                logger.warning("Exception thrown when trying to convert data for {name} to a numpy array for saving to HDF5. Exception message '{msg}'. Attempting to continue saving the data. This might get perilous.".format(name=name, msg=str(e)))
    
        try:
            group.create_dataset(name, data=value)
        except Exception as e:
            logger.error("Exception thrown when trying to save data for {name}. Exception message: '{msg}'.".format(name=name, msg=str(e)))
        
    @if_master 
    def _save(self, root, logger):
        # root is some HDF5 object where a new group will be created and data will be stored
        
        # Generate the unique save name, and generate the HDF5 group
        save_name = Result._get_unique_name(time.time(), self.parameters)
        group = root.create_group(save_name)
        
        # Set the group attributes to the parameters. 
        for parameter, value in self.parameters.items():
            try:
                # Errors are handled here too, just to make sure it saves as much data as it knows how to handle.
                # If something goes wrong here, the user is probably doing something really wrong, but we still catch and log it.
                group.attrs[parameter] = value
            except Exception as e:
                logger.error("Exception thrown when trying to set attribute {name} to {value} in the result data group".format(name=name, value=value))
            
        # Start creating and saving results
        for name, value in self.values.items():
            Result._save_data(group, name, value, logger)
            
class CSVParameterLoader:
    def __init__(self, fname):
        self.fname = fname
        self.parameters = dict()
        
        csv = pd.read_csv(fname)
        for col in csv.columns:
            self.parameters[col] = list(csv[col])
            
        if not assert_parameter_lengths(self.parameters):
            raise ValueError("Parameters do not have the same length")

class ParameterChangelog:
    def __init__(self):
        self.changes = dict()
        self.prev = None
        self.current = None
        
    def _track_parameters(self, parameters: dict[str, typing.Any]):
        self.prev = self.current
        self.current = parameters
        
        if self.prev: # Not the first run
            self.changes = dict()
            for key, value in self.current.items():
                self.changes[key] = (hash(self.current[key]) != hash(self.prev[key]))
        else: # First run
            self.changes = dict()
            for key, value in self.current.items():
                self.changes[key] = True
                
    def changesInclude(self, key_list: list[str]):
        for key in key_list:
            if self.changes[key]:
                return True
        return False
    
    def changesExclude(self, not_key_list: list[str]):
        for key in self.changes.keys():
            if (not key in not_key_list) and self.changes[key]:
                return True
        return False
            

'''
The Simulation class is meant to be inhereted from in the user's code. They 
create a derived class and override some of the methods to tailor the behavior
to their problem. This class allows the following:
 - One time initialization is performed in their constructor (__init__ method).
 - Desired simulation parameter loading (for sweeps) can be performed
   automatically (e.g., with the .basicSweep() command).
 - A template for drawing, running, and post-processing are provided. These
   routines are ran in a fault-tolerant way. Any Python exceptions are caught
   and logged. The Simulation class then moves on to the next simulation
   (if there are more to run).
 - The .run() and .process() methods can return an object of the Result class.
   This stores data that will be automatically saved in the HDF5 format.
'''
class Simulation:
    def __init__(self, logname, working_dir="working_dir"):       
        # Setup logging
        self.logname = logname
        self.logger = Simulation._setup_logging(logname)
            
        # Working directory
        self.working_dir = working_dir
        if_master(make_path_exist)(working_dir)
        
        # Result file
        self.root_result = None
        self._setup_root_result()
    
    @if_master
    def _setup_logging(logname):
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger           
      
    @if_master
    def _setup_root_result(self):
        self.root_result = h5py.File('{logname}.hdf5'.format(logname=self.logname), 'a')
    
    '''
    Core routines that are overridden by the user when they develop their 
    simulations.
    '''
    def draw(self, parameters):
        pass
    
    def run(self, parameters):
        return Result(parameters)
    
    @if_master
    def process(self, result, parameters):
        return result
    
    '''
    Logging wrapper routines to ensure the master process is only logging.
    '''
    @if_master
    def _log_info(self, msg: str):
        self.logger.info(msg)
        
    @if_master
    def _log_warning(self, msg: str):
        self.logger.warning(msg)
        
    @if_master
    def _log_error(self, msg: str):
        self.logger.error(msg)
        
    @if_master
    def _log_critical(self, msg: str):
        self.logger.error(msg)

    '''
    Some predefined functions that run the .draw(), .run(), and .process()
    routines in some fashion, such as a basic parameter sweep based off of 
    a list of parameters. 
    '''
    def basicSweep(self, parameters: dict[str, list[typing.Any]]):
        # Provides a basic way to sweep parameters
        self._log_info("Starting a basic sweep.")
        
        # Check to make sure all parameter lists are the same length
        if assert_parameter_lengths(parameters):
            total_iterations = get_parameter_lengths(parameters)
            
            for iteration in range(total_iterations):
                self._log_info("Starting {iteration}/{total_iterations}".format(iteration=iteration+1, total_iterations=total_iterations))
                
                # So when we are running alot of simulations, we don't want some random error in one of them to kill this entire process
                # We suppress and log it
                drawn = False
                iteration_parameters = get_parameter_iteration(iteration, parameters)

                try:
                    self.draw(iteration_parameters)                    
                    drawn = True
                except Exception as e:
                    self._log_error("Failed drawing {iteration}/{total_iterations} with exception '{exception}'".format(iteration=iteration+1, total_iterations=total_iterations, exception=str(e)))
                else:
                    self._log_info("Finished drawing {iteration}/{total_iterations}".format(iteration=iteration+1, total_iterations=total_iterations))
                
                if drawn:
                    result = None
                    try:
                        result = self.run(iteration_parameters)                    
                    except Exception as e:
                        self._log_error("Failed running {iteration}/{total_iterations} with exception '{exception}'".format(iteration=iteration+1, total_iterations=total_iterations, exception=str(e)))
                    else:
                        self._log_info("Finished running {iteration}/{total_iterations}".format(iteration=iteration+1, total_iterations=total_iterations))
                    
                    # If a result has been returned from the running, then we can process it and allow the user to modify it
                    if is_master() and result is not None:
                        user_result = None
                        try:
                            user_result = self.process(result, iteration_parameters)
                        except Exception as e:
                            self._log_error("Failed processing {iteration}/{total_iterations} with exception '{exception}'".format(iteration=iteration+1, total_iterations=total_iterations, exception=str(e)))
                            result._save(self.root_result, self.logger) # Save what we have, even though the user failed processing it
                        else:
                            # If the returned result is not None, we save it
                            if user_result is not None:
                                user_result._save(self.root_result, self.logger)
                                
            self._log_info("Finished basic sweep")
        else:
            self._log_critical("All parameters in the sweep must have the same length, terminating")
            raise ValueError("Basic sweep failed on all parameters not having uniform length")
            
    