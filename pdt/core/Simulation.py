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
import copy

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
        
    def updateParameters(self, parameters: dict[str, typing.Any]):
        self.prev = copy.deepcopy(self.current)
        self.current = copy.deepcopy(parameters)
        
        if self.prev: # Not the first run
            self.changes = dict()
            for key, value in self.current.items():
                if isinstance(self.current[key], list):
                    self.changes[key] = (hash(tuple(self.current[key])) != hash(tuple(self.prev[key])))
                else:
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
            

class Simulation:
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
    def __init__(self, logname : str, working_dir : str, catch_errors=True):     
        """
        Constructor for the Simulation super class.         

        Parameters
        ----------
        logname : str
            The file name of data and logs to be generated.
        working_dir : str
            The directory to store all file outputs.
        catch_errors : bool, optional
            If True, exceptions will be suppressed and logged. Otherwise, they will
            be allowed to propagate.

        Returns
        -------
        None.

        """
        
        # Working directory
        self.working_dir = working_dir
        if_master(make_path_exist)(working_dir)
        
        # Setup logging
        self.logname = logname
        self.logger = Simulation._setup_logging(logname, working_dir)
        
        # Debug options
        self.catch_errors = catch_errors
    
    @if_master
    def _setup_logging(logname, working_dir):
        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            
            fh = logging.FileHandler("{working_dir}/{logname}.log".format(working_dir=working_dir, logname=logname), mode='a')
            fh.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            
            logger.addHandler(ch)
            logger.addHandler(fh)
        return logger           
    
    def draw(self, parameters : dict[str, typing.Any]) -> None:
        """
        The method to override if you wish to draw your simulation in a separate
        block than your run code.
        
        This is called from all MPI processes.

        Parameters
        ----------
        parameters : dict[str, typing.Any]
            A dictionary of parameters to use in the simulation.

        Returns
        -------
        None.

        """
        pass
    
    def run(self, parameters : dict[str, typing.Any]) -> Result:
        """
        The method to override where you call code that actually runs the simulation.
        
        This is called from all MPI processes.

        Parameters
        ----------
        parameters : dict[str, typing.Any]
            A dictionary of parameters to use in the simulation.

        Returns
        -------
        Result
            A Result object containing whatever simulation data is relevant.
        """
        return Result(parameters)
    
    @if_master
    def process(self, result : Result, parameters : dict[str, typing.Any]) -> Result:
        """
        

        Parameters
        ----------
        result : Result
            The result object returned from the run method.
        parameters : dict[str, typing.Any]
            A dictionary of parameters to use in the simulation.

        Returns
        -------
        Result
            A Result object containing whatever simulation data is relevant.

        """
        return result
    
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
        
    def logInfo(self, msg : str):
        """
        Log a message on the info level.

        Parameters
        ----------
        msg : str
            The message.

        Returns
        -------
        None.

        """
        self._log_info(msg)
        
    def logWarning(self, msg : str):
        """
        Log a message on the warning level.

        Parameters
        ----------
        msg : str
            The message.

        Returns
        -------
        None.

        """
        self._log_warning(msg)
        
    def logError(self, msg : str):
        """
        Log a message on the error level.

        Parameters
        ----------
        msg : str
            The message.

        Returns
        -------
        None.

        """
        self._log_error(msg)

    def oneOff(self, iteration_parameters: dict[str, typing.Any], iteration=1, total_iterations=1) -> typing.Union[Result, None]:
        """
        This method attempts to run a single simulation with the parameters given
        by iteration_parameters. It will call the draw, run, and process methods.
        It will also suppress and log errors if they occur (assuming catch_errors=True).

        Parameters
        ----------
        iteration_parameters : dict[str, typing.Any]
            The parameters to be used in this one-off simulation.
        iteration : int, optional
            If this was called in a larger loop, this can be set to the count 
            of the simulation for logging purposes. The default is 1.
        total_iterations : TYPE, optional
            If this was called in a larger loop, this can be set to the total
            count of the simulation for logging purposes. The default is 1.

        Returns
        -------
        typing.Union[Result, None]
            A Result object if the process routine also returns one. None otherwise
            or in the case of errors.

        """
        
        # The core code that gets called on all processes
        def _one_off(root_result):
            # We suppress and log it
            drawn = False
            
            if self.catch_errors:
                try:
                    self.draw(iteration_parameters)                    
                    drawn = True
                except Exception as e:
                    self._log_error("Failed drawing {iteration}/{total_iterations} with exception '{exception}'".format(iteration=iteration, total_iterations=total_iterations, exception=str(e)))
                else:
                    self._log_info("Finished drawing {iteration}/{total_iterations}".format(iteration=iteration, total_iterations=total_iterations))
                
                if drawn:
                    result = None
                    try:
                        result = self.run(iteration_parameters)                    
                    except Exception as e:
                        self._log_error("Failed running {iteration}/{total_iterations} with exception '{exception}'".format(iteration=iteration, total_iterations=total_iterations, exception=str(e)))
                    else:
                        self._log_info("Finished running {iteration}/{total_iterations}".format(iteration=iteration, total_iterations=total_iterations))
                    
                    # If a result has been returned from the running, then we can process it and allow the user to modify it
                    if result is not None:
                        user_result = None
                        try:
                            user_result = self.process(result, iteration_parameters)
                        except Exception as e:
                            self._log_error("Failed processing {iteration}/{total_iterations} with exception '{exception}'".format(iteration=iteration, total_iterations=total_iterations, exception=str(e)))
                            result._save(root_result, self.logger) # Save what we have, even though the user failed processing it
                        else:
                            # If the returned result is not None, we save it and return it
                            if user_result is not None:
                                user_result._save(root_result, self.logger)
                                return user_result
            else:
                self.draw(iteration_parameters)
                result = self.run(iteration_parameters)
                
                if result is not None:
                    user_result = self.process(result, iteration_parameters)
                    
                    if user_result is not None:
                        user_result._save(root_result, self.logger)
                        return user_result
    
        # We determine whether or not to pass the root_result file to the core code (only should occur on master process)
        if is_master():
            with h5py.File('{working_dir}/{logname}.hdf5'.format(working_dir=self.working_dir, logname=self.logname), 'a') as root_result:
                return _one_off(root_result)
        else:
            return _one_off(None)
        
        # So when we are running alot of simulations, we don't want some random error in one of them to kill this entire process

    def basicSweep(self, parameters: dict[str, list[typing.Any]]):
        """
        This essentially runs multiple oneOff simulations in an automated fashion.

        Parameters
        ----------
        parameters : dict[str, list[typing.Any]]
            A dictionary of parameters, but the item is now a list specifying
            the different parameters to run the simulations with.

        Returns
        -------
        None.

        """
        
        # Provides a basic way to sweep parameters
        self._log_info("Starting a basic sweep.")
        
        # Check to make sure all parameter lists are the same length
        if assert_parameter_lengths(parameters):
            total_iterations = get_parameter_lengths(parameters)
            
            for iteration in range(total_iterations):
                self._log_info("Starting {iteration}/{total_iterations}".format(iteration=iteration+1, total_iterations=total_iterations))
                
                iteration_parameters = get_parameter_iteration(iteration, parameters)
                self.oneOff(iteration_parameters, iteration+1, total_iterations)
                                
            self._log_info("Finished basic sweep")
        else:
            self._log_critical("All parameters in the sweep must have the same length, terminating")

class PreviousResults:
    """
    This class provides functionality to load in previous parameters and the results
    of the corresponding simulations from HDF5 files. 
    """
    
    def __init__(self, logname : str, working_dir : str):
        """
        Constructor for PreviousResults.

        Parameters
        ----------
        logname : str
            See the same variable for the Simulation class.
        working_dir : str
            See the same variable for the Simulation class.

        Returns
        -------
        None.

        """
        
        self.logname = logname
        self.working_dir = working_dir
        
    def getBestParameters(self, fom : str, objective, maximize_objective : bool) -> typing.Union[dict[str, typing.Any], None]:
        """
        This will look at the HDF5 file specified by logname and working_dir.
        It will load in the parameters deemed best by the objective and the fom
        and return them. 

        Parameters
        ----------
        fom : str
            The key for the data determined to be the figure of merit.
        objective : function(object) -> float
            A function that takes the object found in the file matching the key
            denoted by the fom variable and converts it into a float.
        maximize_objective : bool
            Find the parameters that maximize the objective.

        Returns
        -------
        typing.Union[dict[str, typing.Any], None]
            The parameters or None, if they are not found for some reason (i.e.,
            error or file not found).

        """
        
        try: # If the file doesn't exist
            with h5py.File('{working_dir}/{logname}.hdf5'.format(working_dir=self.working_dir, logname=self.logname), 'r') as root_result:
                # Loop through each simulation, and if the fom is present
                best = None
                for sim_name in root_result:
                    # Try to get the figure of merit from the simulation
                    obj_value = None
                    try:
                        obj_value = objective(root_result[sim_name][fom])
                        
                        if best:
                            if maximize_objective:
                                if obj_value > objective(root_result[best][fom]):
                                    best = sim_name
                            else:
                                if obj_value < objective(root_result[best][fom]):
                                    best = sim_name
                        else:
                            best = sim_name
                    except Exception as e:
                        print(e)
                        print("not present in {sim_name}".format(sim_name=sim_name))
                        pass # Not present in that simulation
                
                if best:
                    return dict(root_result[best].attrs)
                else:
                    return None
        except Exception as e:
            return None