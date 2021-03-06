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
import matplotlib.pyplot as plt

from pdt.core.Util import *
from pdt.core.Util import if_master

class Result:
    """
    This class streamlines saving results to HDF5 files. It only does things
    when the master process (i.e., only the master process saves data).
    """
    def __init__(self, parameters: dict[str, typing.Any], **values):
        """
        Constructor for the Result class. It takes in all information that needs
        to be saved.

        Parameters
        ----------
        parameters : dict[str, typing.Any]
            The simulation parameters to be saved as meta-data in the HDF5 file.
        **values : Array-like objects.
            Objects that will be saved as array objects in the HDF5 file.

        Returns
        -------
        None.

        """
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
    """
    A simple helper class for loading parameters from a CSV file into the
    dict[str, list[typing.Any]] format required by Simulation.basicSweep.
    
    Attributes
    ----------
    parameters : dict[str, list[typing.Any]]
        The parameters loaded from the CSV.
    """
    
    def __init__(self, fname : str):
        """
        The constructor of CSVParameterLoader, which also happens to be the only
        method and performs loading into the parameters attribute.

        Parameters
        ----------
        fname : str
            The CSV file's name and path.

        Raises
        ------
        ValueError
            If the columns of the CSV file are not of equal length, an exception
            is raised..

        Returns
        -------
        None.

        """
        self.fname = fname
        self.parameters = dict()
        
        csv = pd.read_csv(fname)
        for col in csv.columns:
            self.parameters[col] = list(csv[col])
            
        if not assert_parameter_lengths(self.parameters):
            raise ValueError("Parameters do not have the same length")

class ParameterChangelog:
    """
    A helper class that can be passed parameters every iteration and determines
    which ones change. This could be useful in determining whether or not to
    redraw the simulation every iteration.
    """
    
    def __init__(self):
        """
        The constructor for ParameterChangelog.

        Returns
        -------
        None.

        """
        self.changes = dict()
        self.prev = None
        self.current = None
        
    def updateParameters(self, parameters: dict[str, typing.Any]):
        """
        This function loads in the new parameters and compares them to what was
        previously passed in. If there were no parameters previously passed in,
        the logic in this function determines all the parameters have changed.

        Parameters
        ----------
        parameters : dict[str, typing.Any]
            The new parameters.

        Returns
        -------
        None.

        """
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
                
    def changesInclude(self, key_list: list[str]) -> bool:
        """
        An 'or' comparison of the changes of the parameters in the key_list.

        Parameters
        ----------
        key_list : list[str]
            The parameters to check the change of.

        Returns
        -------
        bool
            An indication of whether or not atleast one of the parameters in the
            key_list has changed.
        """
        
        for key in key_list:
            if self.changes[key]:
                return True
        return False
    
    def changesExclude(self, not_key_list: list[str]) -> bool:
        """
        An 'or' comparison of the changes of the parameters not in key_list.

        Parameters
        ----------
        not_key_list : list[str]
            The parameters not to check for changes in.

        Returns
        -------
        bool
            An indication of whether or not atleast one of the parameters not
            in the key_list has changed.
        """
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
            
class ConvergenceTest:
    def __init__(self, 
                 sim: Simulation, 
                 convergence_parameters: dict[str, tuple[typing.Union[int, float], typing.Union[int, float]]],
                 fom: str,
                 relative_change=0.01, 
                 absolute_change=None,
                 max_steps=10,
                 certainty_steps=3,
                 interval_generation=np.linspace):
        """
        The constructor for ConvergenceTest. It takes information that dictates
        how convergence testing will be performed.

        Parameters
        ----------
        sim : Simulation
            The simulation to convergence test.
        convergence_parameters : dict[str, tuple[typing.Union[int, float], typing.Union[int, float]]]
            A dictionary of parameter names to convergence test along with a 
            tuple of two elements (lower, higher). This range dictates the range
            over the convergence will be tested. Note that "higher" indicates
            higher expected accuracy. If convergence testing a parameter like
            maximum mesh size, it is perfectly ok to pass in something like
            (1e-6, 1e-9).
        fom : str
            The field to pull out of the Result object returned by the simulation.
            This is what will be used to judge convergence.
        relative_change : float or None, optional
            This parameter specifies how strict the convergence is. When the change
            is < relative_change * np.abs(np.max(foms) - np.min(foms)), convergence
            has been reached. The default is 0.01.
        absolute_change : float or None, optional
            This parameter specifies how strict the convergence is. When the change
            is < absolute_change, convergence has been reached. The default is None.
            Note that relative_change or absolute_change cannot both be None.
        max_steps : int, optional
            The maximum number of steps to divide the ranges provided in the 
            convergence parameters into. The default is 10.
        certainty_steps : int, optional
            The minimum number of recent points visited that meet the convergence
            criteria. The default is 3.
        interval_generation : function of the signature (lower, upper, steps)->array, optional
            The default is np.linspace. Another canidate is np.geomspace. These
            functions dictate how to split up the interval for each parameter
            being swept for convergence.

        Raises
        ------
        ValueError
            Rasied if relative_change is None and absolute_change is None.

        Returns
        -------
        None.

        """
    
        self.sim = sim
        self.convergence_parameters = convergence_parameters
        self.fom = fom
        
        self.relative_change = relative_change
        self.absolute_change = absolute_change
        self.max_steps = max_steps
        self.certainty_steps = certainty_steps
        
        self.interval_generation = interval_generation
        
        if (self.relative_change is None) and (self.absolute_change is None):
            raise ValueError("Note that relative_change must be defined or absolute_change must be defined")
    
    def _has_converged(self, pname, foms):
        if len(foms) > self.certainty_steps:
            # If the last certainty_steps have converged, then we indicate convergence
            # Convergence is defined as changing less than relative_change * |max(fom) - min(fom)|
            
            # Absolute convergence
            converged = True
            for i in range(self.certainty_steps):       
                change = np.abs(foms[-1-i] - foms[-2-i])
                
                # Each of the last certainty_steps must satisfy absolute or relative convergence
                iteration_change = False
                if self.relative_change is not None:
                    relative_change_threshold = self.relative_change * np.abs(np.max(foms) - np.min(foms))
                    iteration_change |= (change < relative_change_threshold)
                
                if self.absolute_change is not None:
                    iteration_change |= (change < self.absolute_change)
                
                converged &= iteration_change
                if not converged:
                    self.sim._log_info("change for {pname} did not satisfy convergence: {change}".format(pname=pname, change=change))
                    break
            
            return converged

        else:
            return False
    
    def _converge_single(self, pname, interval_vals, base_parameters):
        self.sim._log_info("Beginning convergence for {pname}".format(pname=pname))
        parameters = copy.deepcopy(base_parameters)
        
        test_vals = []
        foms = []
        suggested_val = None
        for i, test_val in enumerate(interval_vals):
            parameters[pname] = test_val
            
            result = self.sim.oneOff(parameters)
            fom = result.values[self.fom]
            
            test_vals.append(test_val)
            foms.append(fom)
            
            suggested_val = interval_vals[i-self.certainty_steps]
            
            if self._has_converged(pname, foms):
                self.sim._log_info("{pname} converged at {suggested_val}".format(pname=pname, suggested_val=suggested_val))
                break
            
        return suggested_val, test_vals, foms                
    
    def run(self, base_parameters: dict[str, typing.Any], plot_fname=None) -> dict[str, typing.Union[int, float]]:
        """
        Run the convergence test.

        Parameters
        ----------
        base_parameters : dict[str, typing.Any]
            The parameters to initially assume.
        plot_fname : str or None, optional
            The file prefix to save convergence plots with. The default is None,
            which implies no plots will be generated.

        Raises
        ------
        ValueError
            There are two parameters types capable of being swept: integer and
            floating point numbers. This function will attempt to deduce whether
            to round to the nearest integer. This error will be raised if it
            cannot deduce that.

        Returns
        -------
        resulting_base_parameters : dict[str, typing.Union[int, float]]
            The suggested parameters to use after convergence testing.
            It includes parameters passed in that were not convergence tested,
            also.

        """
        
        self.sim._log_info("Starting convergence testing")
        
        results = dict()
        
        resulting_base_parameters = copy.deepcopy(base_parameters)        
        for pname, interval in self.convergence_parameters.items():
            # Generate the interval
            interval_vals = None
            if isinstance(interval[0], int) and isinstance(interval[1], int):
                # We generate integers
                interval_vals = np.round(self.interval_generation(interval[0], interval[1], self.max_steps))
            elif isinstance(interval[0], float) and isinstance(interval[1], float):
                # We generate floats
                interval_vals = self.interval_generation(interval[0], interval[1], self.max_steps)
            else:
                error_str = "Convergence testing of {pname} on ({start}, {end}) failed because the type of the interval could not be deduced to be a float or int.".format(pname=pname, start=interval[0], end=interval[1])
                self.sim._log_error(error_str)
                raise ValueError(error_str)
            
            suggested_val, test_vals, foms = self._converge_single(pname, interval_vals, resulting_base_parameters)
            results[pname] = suggested_val
            resulting_base_parameters[pname] = suggested_val
            
            if plot_fname is not None:
                plt.figure()
                plt.title("convergence for {pname}".format(pname=pname))
                plt.xlabel("value")
                plt.ylabel("fom")
                plt.scatter(test_vals, foms)
                plt.savefig("{working_dir}/{plot_fname}_{pname}.png".format(working_dir=self.sim.working_dir, plot_fname=plot_fname, pname=pname))
            
        self.sim._log_info("Convergence testing finished with {results}".format(results=results))
            
        return resulting_base_parameters
            
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