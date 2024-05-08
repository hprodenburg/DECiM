"""Part of DECiM. This file contains the fitting classes. Last modified 8 May 2024 by Henrik Rodenburg.

Classes:
RefinementEngine -- class for refinements in general, independent of the GUI
SimpleRefinementEngine -- class for the simple refinement
MultistartEngine -- class for automatic initial guesses
RefinementWindow -- class for the advanced refinement"""

###########
##IMPORTS##
###########

import numpy as np
import scipy.optimize as op

import jax.numpy as jnp
from jax import grad
import optax as ox

import matplotlib as mp
import matplotlib.pyplot as pt
import matplotlib.backends.backend_tkagg as btk
import matplotlib.figure as fg
import matplotlib.animation as anim

import tkinter as tk
import tkinter.ttk as ttk

from ecm_datastructure import dataSet
from ecm_helpers import nearest

######################
##GENERAL REFINEMENT##
######################

class RefinementEngine():
    def __init__(self, function_to_fit, jnp_function_to_fit, data, parameters, parameter_dict, to_refine, high_f, low_f, previous_parameters, opt_module = "SciPy", opt_method = "Nelder-Mead", silent = False, nmaxiter = 500, weighting_scheme = "Unit"):
        """General handling of refinements.
        
        Init arguments (all of which become attributes with the same name):
        function_to_fit -- function to fit; typically Circuit.impedance
        jnp_function_to_fit -- JAX.NumPy function to fit; typically Circuit.jnp_impedance
        data -- dataSet object containing measured data
        parameters -- list of parameters
        parameter_list -- dict of parameters; keys are are indices in self.parameters, values are names (e.g. 'R0', 'n1', ...)
        to_refine -- list of ones and zeros; 1 means that the value in self.parameters at the same index should be refined, 0 means that it should not; does not work with optax.adam
        high_f -- upper frequency bound
        low_f -- lower frequency bound
        previous_parameters -- list of lists of parameters
        opt_module -- string, module from which to import optimizer; only option for now is 'SciPy'
        opt_method -- string, optimizer to use
        nmaxiter -- maximum number of iterations per refineable parameter
        weighting_scheme -- weighting scheme, string; one of 'Unit', 'Observed modulus', 'Calculated modulus', 'Observed proportional', and 'Calculated proportional'
        
        Attributes (other):
        module_dict -- dict of modules and corresponding refinement methods; keys are strings, values are methods in this class.
        l_idx, h_idx -- indices of the lower and upper frequency limits, respectively; created during refinement
        
        Methods:
        refine_solution -- refine the solution and update self.parameters
        refine_solution_scipy -- refine the solution with scipy.optimize.minimize and update self.parameters
        refine_solution_optax -- refine the solution with optax and update self.parameters
        error_function -- function that returns the error to be minimized by self.refine_solution"""
        self.function_to_fit = function_to_fit
        self.jnp_function_to_fit = jnp_function_to_fit
        self.data = data
        self.parameters = parameters
        self.parameter_dict = parameter_dict
        self.to_refine = to_refine
        self.high_f = high_f
        self.low_f = low_f
        self.previous_parameters = previous_parameters
        self.opt_module = opt_module
        self.opt_method = opt_method
        self.silent = silent
        self.nmaxiter = nmaxiter
        self.weighting_scheme = weighting_scheme
        
        self.module_dict = {"SciPy": self.refine_solution_scipy, "Optax": self.refine_solution_optax}
        
    def refine_solution(self):
        """Choose the correct refinement method based on self.opt_module."""
        if self.opt_module in self.module_dict:
            return self.module_dict[self.opt_module]()
        else:
            print("Refinement error: module {:s} not defined.".format(self.opt_module))
    
    def refine_solution_scipy(self):
        """Apply boundaries, update the parameter history and minimize self.error_function using scipy.optimize.minimize."""
        #Frequency limits
        self.h_idx, self.l_idx = nearest(self.high_f, self.data.freq), nearest(self.low_f, self.data.freq)
        #Boundary determination & number of refineable parameters (for maximum iterations)
        ref_par_count = 0
        boundaries = []
        for t in range(len(self.to_refine)):
            if self.to_refine[t] == 0: #Parameter not included in refinement
                boundaries.append((self.parameters[t], self.parameters[t])) #Constrain parameter to its present value
            elif self.to_refine[t] == 1: #Parameter included in refinement
                ref_par_count += 1
                if self.parameter_dict[t][0] in ["n", "b", "g"]:
                    boundaries.append((0, 1)) #Refine the parameter (n, b, g bounded between 0 and 1)
                else:  
                    boundaries.append((1e-30, 1e15)) #Refine the parameter (other parameters bounded between 0 and 1e15)
            else:
                print("Warning: forbidden refinement option. Constraining parameter {:s}.".format(self.parameters[t]))
                boundaries.append((self.parameters[t], self.parameters[t]))
        #Save previous refinement
        self.previous_parameters.append(self.parameters)
        #Refinement
        self.iter = 0
        opt_res = op.minimize(self.error_function, np.array(self.parameters), method = self.opt_method, options = {"maxiter": self.nmaxiter*ref_par_count}, bounds = boundaries, callback = self.optimisation_callback) #Minimise the error
        self.parameters = opt_res["x"] #Save new parameters
        
    def refine_solution_optax(self):
        """Apply boundaries, update the parameter history and use the optax.adam optimizer to find a solution."""
        #Frequency limits
        self.h_idx, self.l_idx = nearest(self.high_f, self.data.freq), nearest(self.low_f, self.data.freq)
        #Number of refineable parameters (for maximum iterations)
        ref_par_count = len(self.parameters)
        #Save previous refinement
        self.previous_parameters.append(self.parameters)
        #Convert sliced (frequency boundaries applied) NumPy arrays to JAX.NumPy arrays
        self.jnp_freq = jnp.asarray(self.data.freq[self.l_idx:self.h_idx])
        self.jnp_real = jnp.asarray(self.data.real[self.l_idx:self.h_idx])
        self.jnp_imag = jnp.asarray(self.data.imag[self.l_idx:self.h_idx])
        #Convert parameters array to JAX.NumPy array
        jnp_parameters = jnp.asarray(self.parameters)
        #Solver initialization
        solvers = {"Adam": ox.adam}
        solver = solvers[self.opt_method](learning_rate = 0.005)
        opt_state = solver.init(jnp_parameters)
        #Solution
        self.iter = 0
        for i in range(int(ref_par_count*self.nmaxiter/10)):
            gradient = grad(self.error_function_jnp)(jnp_parameters)
            updates, opt_state = solver.update(gradient, opt_state, jnp_parameters)
            jnp_parameters = ox.apply_updates(jnp_parameters, updates)
            #self.optimisation_callback(jnp_parameters)
        self.parameters = np.asarray(jnp_parameters)
        
    def error_function(self, params):
        """Function to be minimized in the refinement. Weighting schemes are applied here.
        
        Arguments:
        self
        params -- list of fit parameters
        
        Returns:
        NumPy array; sum of the squared differences in the impedance; this is done separately for the real and imaginary components, which are then added."""
        w_re, w_im = 1, 1 #Unit weighting is default
        if self.weighting_scheme == "Calculated modulus":
            w_re = np.sqrt(np.real(self.function_to_fit(params, self.data.freq))**2 + np.imag(self.function_to_fit(params, self.data.freq))**2)
            w_im = w_re
        elif self.weighting_scheme == "Observed modulus":
            w_re = np.sqrt(self.data.real**2 + self.data.imag**2)
            w_im = w_re
        elif self.weighting_scheme == "Calculated proportional":
            w_re = 1/(np.real(self.function_to_fit(params, self.data.freq))**2)
            w_im = 1/(np.imag(self.function_to_fit(params, self.data.freq))**2)
        elif self.weighting_scheme == "Observed proportional":
            w_re = 1/(self.data.real**2)
            w_im = 1/(self.data.imag**2)
        if self.l_idx == 0 and self.h_idx == -1:
            return sum(w_re*(np.real(self.function_to_fit(params, self.data.freq)) - self.data.real)**2) + sum(w_im*(np.imag(self.function_to_fit(params, self.data.freq)) - self.data.imag)**2)
        #If limits are set, return the limited result (with weights being taken care of first)
        if self.weighting_scheme not in ["", "Unit"]:
            w_re, w_im = w_re[self.l_idx:self.h_idx], w_im[self.l_idx:self.h_idx]
        return sum(w_re*(np.real(self.function_to_fit(params, self.data.freq[self.l_idx:self.h_idx])) - self.data.real[self.l_idx:self.h_idx])**2) + sum(w_im*(np.imag(self.function_to_fit(params, self.data.freq[self.l_idx:self.h_idx])) - self.data.imag[self.l_idx:self.h_idx])**2)
        
    def error_function_jnp(self, params):
        """Function to be minimized in the refinement. Weighting schemes are applied here.
        
        Arguments:
        self
        params -- list of fit parameters
        
        Returns:
        JAX.NumPy array; sum of the squared differences in the impedance; this is done separately for the real and imaginary components, which are then added."""
        w_re, w_im = 1, 1 #Unit weighting is default
        if self.weighting_scheme == "Calculated modulus":
            w_re = jnp.sqrt(np.real(self.jnp_function_to_fit(params, self.jnp_freq))**2 + np.imag(self.jnp_function_to_fit(params, self.jnp_freq))**2)
            w_im = w_re
        elif self.weighting_scheme == "Observed modulus":
            w_re = jnp.sqrt(self.jnp_real**2 + self.jnp_imag**2)
            w_im = w_re
        elif self.weighting_scheme == "Calculated proportional":
            w_re = 1/(jnp.real(self.jnp_function_to_fit(params, self.jnp_freq))**2)
            w_im = 1/(jnp.imag(self.jnp_function_to_fit(params, self.jnp_freq))**2)
        elif self.weighting_scheme == "Observed proportional":
            w_re = 1/(self.jnp_real**2)
            w_im = 1/(self.jnp_imag**2)
        if self.l_idx == 0 and self.h_idx == -1:
            return sum(w_re*(np.real(self.jnp_function_to_fit(params, self.jnp_freq)) - self.jnp_real)**2) + sum(w_im*(np.imag(self.jnp_function_to_fit(params, self.jnp_freq)) - self.jnp_imag)**2)
        #If limits are set, return the limited result (with weights being taken care of first)
        if self.weighting_scheme not in ["", "Unit"]:
            w_re, w_im = w_re, w_im
        return jnp.sum(w_re*(np.real(self.jnp_function_to_fit(params, self.jnp_freq)) - self.jnp_real)**2) + jnp.sum(w_im*(np.imag(self.jnp_function_to_fit(params, self.jnp_freq)) - self.jnp_imag)**2)
        
    def optimisation_callback(self, params):
        """Print the output of the refinement function to the command line for each iteration."""
        if self.silent:
            return
        out_str = "Iteration " + str(self.iter) + " | "
        for p in self.parameter_dict:
            out_str += self.parameter_dict[p] + ": " + "{:5g} | ".format(params[p])
        print(out_str)
        self.iter += 1


#####################
##SIMPLE REFINEMENT##
#####################

class SimpleRefinementEngine():
    def __init__(self, params, data, min_impedance_function, bounded = True):
        """Handling of the simple refinement.
        
        Init arguments:
        params -- list of fit parameters
        data -- dataSet containing measured data (frequency limits should be applied BEFORE calling __init__)
        min_impedance_function -- impedance function used in the optimization
        
        Keyword arguments:
        bounded -- Boolean, True if bounds are to be applied to the element values, False otherwise
        
        Attributes:
        input_params -- list of input parameters
        output_params -- list of output parameters
        data -- dataSet containing measured data
        min_impedance_function -- impedance function used in the minimization
        bounded -- Boolean, True if bounds are to be applied to the element values, False otherwise
        
        Methods:
        errorFunction -- function to be minimized
        minRefinement -- complete optimization method"""
        self.input_params = params #Parameters belonging to the circuit elements
        self.output_params = params
        self.data = data #Raw data, taking the form of a dataSet object (ecm_datastructure). If a limited frequency range is desired, the truncated data should be passed to the SimpleRefinementEngine.
        self.min_impedance_function = min_impedance_function #The function used to calculate the impedance (ecm_circuits)
        self.bounded = bounded

    def errorFunction(self, parameters):
        """Function to be minimized.
        
        Arguments:
        self
        parameters: list of fit parameters
        
        Returns:
        NumPy array; sum of the squared differences in the impedance; this is done separately for the real and imaginary components, which are then added."""
        return sum((np.real(self.min_impedance_function(parameters, self.data.freq)) - self.data.real)**2) + sum((np.imag(self.min_impedance_function(parameters, self.data.freq)) - self.data.imag)**2)

    def minRefinement(self):
        """Parameter optimization function; uses the Nelder-Mead (simplex) method and unit weighting to calculate the optimal parameters."""
        bound_list = []
        for p in self.input_params:
            bound_list.append((0, 1e15))
        if self.bounded:
            opt_res = op.minimize(self.errorFunction, np.array(self.input_params), method="Nelder-Mead", options = {"maxiter": 10000}, bounds = bound_list)
        else:
            opt_res = op.minimize(self.errorFunction, np.array(self.input_params), method="Nelder-Mead", options = {"maxiter": 10000})
        self.output_params = opt_res["x"]

############################
##MULTISTART INITIAL GUESS##
############################

class MultistartEngine():
    def __init__(self, function_to_fit, jnp_function_to_fit, data, parameters, parameter_dict, previous_parameters, opt_module = "SciPy", opt_method = "Nelder-Mead", silent = True, nmaxiter = 100, weighting_scheme = "Unit", starts_per_par = 3, nmaxstarts = 30):
        """Multistart approach for global optimization; used for automatic initial guesses.
        
        Init arguments (all of which become attributes with the same name):
        function_to_fit -- function to fit; typically Circuit.impedance
        jnp_function_to_fit -- JAX.NumPy function to fit; typically Circuit.jnp_impedance
        data -- dataSet object containing measured data
        parameters -- list of parameters
        parameter_list -- dict of parameters; keys are are indices in self.parameters, values are names (e.g. 'R0', 'n1', ...)
        previous_parameters -- list of lists of parameters
        opt_module -- string, module from which to import optimizer; only option for now is 'SciPy'
        opt_method -- string, optimizer to use
        nmaxiter -- maximum number of iterations per refineable parameter in each RefinementEngine
        weighting_scheme -- weighting scheme, string; one of 'Unit', 'Observed modulus', 'Calculated modulus', 'Observed proportional', and 'Calculated proportional'
        starts_per_par -- number of starting positions generated per parameter; default 3
        nmaxstarts -- maximum number of RefinementEngines started by the MultistartEngine; default 30
        
        Other attributes:
        to_refine -- list of ones and zeros; 1 means that the value in self.parameters at the same index should be refined, 0 means that it should not; does not work with optax.adam
        high_f -- upper frequency bound
        low_f -- lower frequency bound
        
        Methods:
        initial_value -- for a given element, randomly generate a reasonable starting value
        initial_parameters -- generate reasonable starting values for all parameters; return a new array
        generate_solution -- generate random starting positions, refine each, and choose the best solution"""
        self.function_to_fit = function_to_fit
        self.jnp_function_to_fit = jnp_function_to_fit
        self.data = data
        self.parameters = parameters
        self.parameter_dict = parameter_dict
        self.to_refine = list(np.ones(len(parameters)))
        self.high_f = max(self.data.freq)
        self.low_f = min(self.data.freq)
        self.previous_parameters = previous_parameters
        self.opt_module = opt_module
        self.opt_method = opt_method
        self.silent = silent
        self.nmaxiter = nmaxiter
        self.weighting_scheme = weighting_scheme
        self.starts_per_par = starts_per_par
        self.nmaxstarts = nmaxstarts
        
    def initial_value(self, parameter_tag_or_name):
        """For any valid parameter tag ('R', 'C', 'L', ..., 'H'), generate a reasonable starting value.
        Reasonable starting values are exponentially (base 10) distributed, covering a wide range of starting values.
        
        Arguments:
        self
        parameter_tag_or_name -- name or tag of parameter
        
        Returns:
        float; reasonable initial value"""
        random_number = np.random.random()
        par_id = parameter_tag_or_name[0]
        if par_id in "ROSGH":
            return min(10**(10*random_number), 10**(np.log10(max(self.data.real)))*random_number)
        elif par_id in "CLQmt":
            return 10**(-12*random_number)
        elif par_id in "kl":
            return 10**(6*random_number)
        elif par_id in "nbg":
            return random_number
            
    def initial_parameters(self):
        """Generate a new array of parameters, with random starting values according to self.initial_value.
        
        Arguments:
        self
        
        Returns:
        NumPy array of parameter values"""
        new_params = np.zeros(len(self.parameters))
        for idx in self.parameter_dict:
            new_params[idx] = self.initial_value(self.parameter_dict[idx])
        return new_params
        
    def generate_solution(self):
        """Fit the model to the data from all starting positions and save the best set of parameters as self.parameters."""
        engines = []
        scores = []
        for i in range(min([len(self.parameters)*self.starts_per_par, self.nmaxstarts])):
            engines.append(RefinementEngine(self.function_to_fit, self.jnp_function_to_fit, self.data, self.initial_parameters(), self.parameter_dict, self.to_refine, self.high_f, self.low_f, self.previous_parameters, opt_module = self.opt_module, opt_method = self.opt_method, silent = self.silent, nmaxiter = self.nmaxiter, weighting_scheme = self.weighting_scheme))
            engines[-1].refine_solution()
            scores.append(engines[-1].error_function(engines[-1].parameters))
        best_idx = scores.index(min(scores))
        self.previous_parameters.append(self.parameters)
        self.parameters = engines[best_idx].parameters

#######################
##ADVANCED REFINEMENT##
#######################

class RefinementWindow(tk.Toplevel):
    def __init__(self, function_to_fit, jnp_function_to_fit, element_list, initial_parameters, data):
        """Advanced refinement window.
        
        Init arguments:
        function_to_fit -- impedance function to be used in the optimization
        element_list -- list of all elements in the Circuit
        initial_parameters -- list of fit parameters
        data -- dataSet containing measured data (no frequency limits!)
        
        Attributes (excluding UI elements):
        width -- window width
        height -- window height
        
        function_to_fit -- impedance function to be used in the optimization
        element_list -- list of all elements in the Circuit
        data -- dataSet containing measured data (no frequency limits!)
        initial_parameters -- list initial fit parameters
        refined_parameters -- list refined fit parameters
        parameter_history -- list of lists of previous parameter values
        parameter_dict -- dict containing parameter names as keys and corresponding indices in initial_parameters and refined_parameters as values
        flipped_parameter_dict -- dict containing indices as values and parameter names as keys
        parameter_list -- list of parameter names
        refinement_accepted -- bool, indicates if the refinement result has been accepted by the user
        limit_visualisation -- bool, indicates whether the frequency limits should be displayed
        silent_optimisation -- bool, indicates whether the callback function should be used during the refinement (True if it should not be used)
        l_idx -- index of lowest-frequency point in data
        h_idx -- index of highest-frequency point in data
        
        Methods:
        make_UI -- initialize the UI; makes the following UI elements, which are attributes: upper_frame, lower_frame, tick_frame, limit_frame, residuals_frame, conclude_frame
        make_tick_frame -- makes the set of tk.Checkbuttons used for parameter selection and their associated labels
        make_limit_frame -- makes the tk.Entries for setting the frequency limits and their associated labels, as well as the weighting scheme dropdown
        make_residuals_frame -- makes the plot canvas
        make_conclude_frame -- create the result label and the buttons used to control the refinement outcome
        set_param_labels -- set the values of StringVars that display the parameters' current values and units
        toggle_lim_vis -- toggle frequency limit visualization on/off
        update_residuals -- clear and redraw the plot canvas
        error_function -- function to be minimized
        display_error -- display the current reduced sum of the squares
        optimisation_callback -- print refinement status to the command line
        refine_solution -- perform the minimization of error_function
        previous_refinement -- change the parameters back to what they were before the latest refinement
        reset_parameters -- change the parameters back to what they were when the refinement window was launched
        accept_refinement -- close the window and change the model parameters to the refinement result
        reject_refinement -- close the window and do not change the model parameters
        flip_parameter_values_keys -- flip self.parameter_dict keys and values
        toggle_all_parameters -- toggle all tick boxes on/off
        optimizer_tick_correction -- disable or enable tick boxes depending on chosen optimizer"""
        super().__init__()
        self.title("Equivalent circuit model refinement")
        self.width = int(self.winfo_screenwidth()*0.85)
        self.height = int(self.winfo_screenheight()*0.85)
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        
        #Variables
        self.function_to_fit = function_to_fit
        self.jnp_function_to_fit = jnp_function_to_fit
        self.element_list = element_list
        self.parameter_dict = {}
        indices = []
        for p in self.element_list: #Tie parameter names to parameter array indices and count the parameters
            self.parameter_dict[p.name] = p.idx
            indices.append(p.idx)
            if p.tag in "QSOGH":
                indices.append(p.idx2)
                if p.tag == "Q":
                    self.parameter_dict["n" + str(p.number)] = p.idx2
                if p.tag == "O":
                    self.parameter_dict["k" + str(p.number)] = p.idx2
                if p.tag == "S":
                    self.parameter_dict["l" + str(p.number)] = p.idx2
                if p.tag == "G":
                    self.parameter_dict["m" + str(p.number)] = p.idx2
                if p.tag == "H":
                    indices.append(p.idx3)
                    indices.append(p.idx4)
                    self.parameter_dict["t" + str(p.number)] = p.idx2
                    self.parameter_dict["b" + str(p.number)] = p.idx3
                    self.parameter_dict["g" + str(p.number)] = p.idx4
        self.flip_parameter_values_keys()
        parcount = max(indices) + 1
        self.parameter_list = list(np.zeros(parcount))
        for p in self.parameter_dict:
            self.parameter_list[self.parameter_dict[p]] = p
        self.initial_parameters = initial_parameters
        self.refined_parameters = initial_parameters[:parcount] #Need to shrink array to make refinement work
        self.parameter_history = [initial_parameters[:parcount]]
        self.data = data
        self.refinement_accepted = False
        self.limit_visualisation = False
        self.silent_optimisation = False
        
        #Frequency limits (indices)
        self.l_idx = 0
        self.h_idx = -1
        
        #Put the UI together
        self.make_UI()
    
    def make_UI(self):
        """Make the RefinementWindow UI. First divide the window into upper and lower halves (upper_frame, lower_frame), then put tick_frame, limit_frame, residuals_frame and conclude_frame on the lower ttk.Frame."""
        #Two frames for vertical division
        self.upper_frame = ttk.Frame(self)
        self.upper_frame.pack(side = tk.TOP, anchor = tk.CENTER, fill = tk.BOTH, expand = tk.YES)
        self.lower_frame = ttk.Frame(self)
        self.lower_frame.pack(side = tk.TOP, anchor = tk.CENTER, fill = tk.BOTH, expand = tk.YES)
        
        #Four functional frames: one for the parameter tick boxes
        self.tick_frame = ttk.Frame(self.lower_frame)
        self.make_tick_frame()
        self.tick_frame.pack(side = tk.LEFT, anchor = tk.W, padx = 50, fill = tk.X, expand = tk.YES)
        #One for the frequency limits
        self.limit_frame = ttk.Frame(self.lower_frame)
        self.make_limit_frame()
        self.limit_frame.pack(side = tk.LEFT, anchor = tk.W, padx = 50, fill = tk.X, expand = tk.YES)
        #One for the residuals
        self.residuals_frame = ttk.Frame(self.upper_frame)
        self.make_residuals_frame()
        self.residuals_frame.pack(side = tk.TOP, anchor = tk.W, fill = tk.BOTH, expand = tk.YES)
        #One for the accept and reject buttons
        self.conclude_frame = ttk.Frame(self.lower_frame)
        self.make_conclude_frame()
        self.conclude_frame.pack(side = tk.LEFT, anchor = tk.W, padx = 50, fill = tk.X, expand = tk.YES)
        
    def make_tick_frame(self):
        """Put tk.Checkbutton objects on the tick_frame. First make lists to hold:
        -- the Checkbuttons themselves: tick_boxes
        -- the tick states (tk.IntVar with 0 or 1 for off or on): tick_states
        -- a list of parameter labels (tk.Label): parameter_labels
        -- a list of parameter values (tk.StringVar): parameter_values
        Then make a Checkbutton and Label for every parameter in parameter_list and pack them onto either tick_box_frame or tick_label_frame.
        Finally, call set_param_labels to update the parameter labels' StringVars."""
        self.tick_boxes = [] #List of tick boxes. The states of these boxes determine which parameters are refined
        self.tick_states = [] #Saves the on/off states of the tick boxes
        self.parameter_labels = [] #List of labels to display current values
        self.parameter_values = [] #List of StringVars to hold parameter values
        self.tick_header_frame = ttk.Frame(self.tick_frame)
        self.tick_header_frame.pack(side = tk.TOP, anchor = tk.W)
        self.tick_label = tk.Label(self.tick_header_frame, text = "Parameters to refine")
        self.tick_label.pack(side = tk.LEFT, anchor = tk.W)
        self.toggle_all_button = tk.Button(self.tick_header_frame, text = "Toggle all", command = self.toggle_all_parameters)
        self.toggle_all_button.pack(side = tk.RIGHT, anchor = tk.W)
        self.tick_box_frame = ttk.Frame(self.tick_frame) #The frame is subdivided into two colums. One for boxes...
        self.tick_box_frame.pack(side = tk.LEFT, anchor = tk.W)
        self.tick_label_frame = ttk.Frame(self.tick_frame) #... and one for labels.
        self.tick_label_frame.pack(side = tk.LEFT, anchor = tk.W)
        for param in self.parameter_list: #A tick box and label are made for every parameter.
            self.tick_states.append(tk.IntVar(self))
            self.tick_boxes.append(tk.Checkbutton(self.tick_box_frame, text = param, variable = self.tick_states[-1]))
            self.parameter_values.append(tk.StringVar(self))
            self.parameter_labels.append(tk.Label(self.tick_label_frame, textvariable = self.parameter_values[-1]))
            if str(param)[0] in "RLCQnOSGklmHtbg": #Only pack real, meaningful parameters.
                self.tick_boxes[-1].pack(side = tk.TOP, anchor = tk.W, padx = 30)
                self.parameter_labels[-1].pack(side = tk.TOP, anchor = tk.E)
        self.set_param_labels()
        
    def make_limit_frame(self):
        """Create tk.Entries and tk.Labels for setting the refinement frequency limits, as well as a tk.OptionMenu from which the weighting schemes can be chosen and a tk.OptionMenu from which the optimizer can be chosen."""
        #Upper limit
        self.high_lim_label = tk.Label(self.limit_frame, text = "Upper frequency limit (Hz)")
        self.high_lim_label.pack(side = tk.TOP, anchor = tk.W)
        self.high_lim_value = tk.StringVar(self, str(max(self.data.freq)))
        self.high_lim_input = tk.Entry(self.limit_frame, textvariable = self.high_lim_value)
        self.high_lim_input.pack(side = tk.TOP, anchor = tk.W)
        #Lower limit
        self.low_lim_label = tk.Label(self.limit_frame, text = "Lower frequency limit (Hz)")
        self.low_lim_label.pack(side = tk.TOP, anchor = tk.W)
        self.low_lim_value = tk.StringVar(self, str(min(self.data.freq)))
        self.low_lim_input = tk.Entry(self.limit_frame, textvariable = self.low_lim_value)
        self.low_lim_input.pack(side = tk.TOP, anchor = tk.W)
        #Limit visualisation
        self.toggle_limit_lines_button = tk.Button(self.limit_frame, text = "Limit visualisation on/off", command = self.toggle_lim_vis)
        self.toggle_limit_lines_button.pack(side = tk.TOP, anchor = tk.W)
        #Weighting
        self.chosen_weighting = tk.StringVar()
        self.weighting_frame = ttk.Frame(self.limit_frame)
        self.weighting_frame.pack(side = tk.BOTTOM, anchor = tk.W)
        self.weighting_label = tk.Label(self.weighting_frame, text = "Weighting scheme: ")
        self.weighting_label.pack(side = tk.LEFT, anchor = tk.W)
        self.weighting_dropdown = tk.OptionMenu(self.weighting_frame, self.chosen_weighting, "Unit")
        self.weighting_dropdown.pack(side = tk.RIGHT, anchor = tk.W)
        self.expanded_weighting_schemes = ["Calculated modulus", "Observed modulus", "Calculated proportional", "Observed proportional"]
        for w in self.expanded_weighting_schemes:
            self.weighting_dropdown["menu"].add_command(label = w, command = tk._setit(self.chosen_weighting, w))
        #Optimizers
        self.chosen_module_and_optimizer = tk.StringVar()
        self.chosen_module_and_optimizer.trace("w", self.optimizer_tick_correction) #Disable tickboxes with Optax
        self.optimizer_frame = ttk.Frame(self.limit_frame)
        self.optimizer_frame.pack(side = tk.BOTTOM, anchor = tk.W)
        self.optimizer_label = tk.Label(self.optimizer_frame, text = "Optimizer: ")
        self.optimizer_label.pack(side = tk.LEFT, anchor = tk.W)
        self.optimizer_dropdown = tk.OptionMenu(self.optimizer_frame, self.chosen_module_and_optimizer, "", command = self.optimizer_tick_correction)
        self.optimizer_dropdown["menu"].delete(0, "end")
        self.optimizer_dropdown.pack(side = tk.RIGHT, anchor = tk.W)
        self.expanded_optimizers = ["SciPy: Nelder-Mead", "Optax: Adam"]
        for o in self.expanded_optimizers:
            self.optimizer_dropdown["menu"].add_command(label = o, command = tk._setit(self.chosen_module_and_optimizer, o))
        
    def make_residuals_frame(self):
        """Create the plotting canvas, the plots on it and the toolbar of the residuals frame."""
        #Make the figure
        self.fig = fg.Figure(figsize = (int(self.width/41), int(self.height/41)), dpi = 82) #Make figure
        self.fig, (self.residuals, self.nyquist, self.bode_amp) = pt.subplots(ncols = 3, nrows = 1)
        self.bode_phase = self.bode_amp.twinx() #Add phase
        self.canvas = btk.FigureCanvasTkAgg(self.fig, self.residuals_frame) #Create canvas on which to draw the figure
        self.canvas.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True) #Place the canvas on the frame
        self.fig.subplots_adjust(wspace = 0.3) #Space plots further apart
        #Add canvas controls (zoom, pan, etc.)
        self.toolbar = btk.NavigationToolbar2Tk(self.canvas, self.residuals_frame)
        self.toolbar.update()
        #Update the figure to set limits and draw data
        self.update_residuals()
        
    def make_conclude_frame(self):
        """Create the result text and the refine, undo, reset, and accept buttons."""
        #Result label
        self.result_text = tk.StringVar(self)
        self.display_error()
        self.result_label = tk.Label(self.conclude_frame, textvariable = self.result_text)
        self.result_label.pack(side = tk.TOP, anchor = tk.E)
        #Refine button
        self.refine_text = tk.StringVar(self)
        self.refine_text.set("Refine solution")
        self.refine_button = tk.Button(self.conclude_frame, textvariable = self.refine_text, command = self.refine_solution)
        self.refine_button.pack(side = tk.TOP, anchor = tk.E)
        #Undo button
        self.undo_button = tk.Button(self.conclude_frame, text = "Undo refinement", command = self.previous_refinement)
        self.undo_button.pack(side = tk.TOP, anchor = tk.E)
        #Reset button
        self.reset_button = tk.Button(self.conclude_frame, text = "Reset parameters", command = self.reset_parameters)
        self.reset_button.pack(side = tk.TOP, anchor = tk.E)
        #Accept button
        self.accept_button = tk.Button(self.conclude_frame, text = "Accept and close", command = self.accept_refinement)
        self.accept_button.pack(side = tk.TOP, anchor = tk.E)
        #Reject button
        self.accept_button = tk.Button(self.conclude_frame, text = "Reject and close", command = self.reject_refinement)
        self.accept_button.pack(side = tk.TOP, anchor = tk.E)
        
    def set_param_labels(self):
        """Set the labels of the parameters on the tick_frame."""
        for p in self.parameter_dict:
            units = "" #Stays this way for n, b, g
            if str(p)[0] in ["R", "O", "S", "G", "H"]:
                units = " Ohm"
            if str(p)[0] == "C":
                units = " F"
            if str(p)[0] == "Q":
                units = " Fs^(n-1)"
            if str(p)[0] == "L":
                units = " H"
            if str(p)[0] in ["k", "l", "m"]:
                units = " s^(1/2)"
            if str(p)[0] == "t":
                units = " s^(b*g)"
            if str(p)[0] in "RLCQnOSGklmHtbg":
                self.parameter_values[self.parameter_dict[p]].set(p + " = {:5g}".format(self.refined_parameters[self.parameter_dict[p]]) + units)
    
    def toggle_lim_vis(self):
        """Toggle the frequency limits visualization in the plots on or off."""
        self.limit_visualisation = not self.limit_visualisation
        self.update_residuals()
    
    def update_residuals(self):
        """Update the plots on the residuals frame: clear them and redraw the canvas."""
        #Clear subplots
        self.residuals.cla()
        self.nyquist.cla()
        self.bode_amp.cla()
        self.bode_phase.cla()
        #Restore settings
        self.residuals.set_title("Residuals")
        self.residuals.set_xlabel("Frequency (Hz)")
        self.residuals.set_xscale("log")
        self.residuals.set_ylabel("(Z$_{model}$ - Z$_{data}$)/Z$_{data}$")
        self.residuals.set_yscale("linear")
        self.nyquist.set_xlabel("Re[Z] ($\Omega$)")
        self.nyquist.set_xscale("linear")
        self.nyquist.set_ylabel("-Im[Z] ($\Omega$)")
        self.nyquist.set_yscale("linear")
        self.bode_amp.set_xlabel("Frequency (Hz)")
        self.bode_amp.set_xscale("log")
        self.bode_amp.set_ylabel("|Z| ($\Omega$)")
        self.bode_amp.set_yscale("log")
        self.bode_amp.yaxis.label.set_color("#2222A0")
        self.bode_amp.tick_params(axis = "y", colors = "#2222A0")
        self.bode_phase.set_ylabel("$\phi$ ($^\circ)$")
        self.bode_phase.yaxis.label.set_color("#229950")
        self.bode_phase.tick_params(axis = "y", colors = "#229950")
        self.bode_phase.yaxis.tick_right()
        self.bode_phase.yaxis.set_label_position("right")
        #Compute solution
        d_solution = self.function_to_fit(self.refined_parameters, self.data.freq)
        #Plot the residuals
        real_res = (np.real(d_solution) - self.data.real)/self.data.real
        imag_res = (np.imag(d_solution) - self.data.imag)/self.data.imag
        self.residuals.plot(self.data.freq, real_res, marker = ".", linestyle = "-", linewidth = 1.5, markersize = 6, fillstyle = "full", color = "#FFA000", markeredgecolor = "#000000", markeredgewidth = 1, label = "Re[Z]")
        self.residuals.plot(self.data.freq, imag_res, marker = "s", linestyle = "-", linewidth = 1.5, markersize = 6, fillstyle = "full", color = "#00A0FF", markeredgecolor = "#000000", markeredgewidth = 1, label = "Im[Z]")
        #Plot the zero lines
        self.residuals.plot(self.data.freq, np.zeros(len(self.data.freq)), linestyle = "-", marker = "None", linewidth = 1, color = "#000000")
        self.residuals.plot(self.data.freq, np.zeros(len(self.data.freq)), linestyle = "-", marker = "None", linewidth = 1, color = "#000000")
        #Update the limits
        self.residuals.set_xlim(left = min(self.data.freq), right = max(self.data.freq))
        rlim = max(abs(max(real_res)) + 0.05*abs(max(real_res)), abs(min(real_res)) + 0.05*abs(min(real_res)), abs(max(imag_res)) + 0.05*abs(max(imag_res)), abs(min(imag_res)) + 0.05*abs(min(imag_res)))
        self.residuals.set_ylim(top = rlim, bottom = -rlim)
        #Update the complex plane
        sim_freq = 10**(np.linspace(np.log10(self.data.freq[0]), np.log10(self.data.freq[-1]), 500))
        s_solution = self.function_to_fit(self.refined_parameters, sim_freq)
        sim_re = np.real(s_solution)
        sim_im = -np.imag(s_solution)
        self.nyquist.plot(self.data.real, -self.data.imag, linestyle = "None", marker = ".", markersize = 6, fillstyle = "full", color = "#A0A0A0", markeredgecolor = "#000000", markeredgewidth = 1)
        self.nyquist.plot(sim_re, sim_im, linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444")
        #Update the limits
        self.nyquist.set_xlim(left = min(self.data.real) - 0.05*max(self.data.real), right = 1.05*max(self.data.real))
        self.nyquist.set_ylim(bottom = min(-self.data.imag) - 0.05*max(self.data.real), top = 1.05*max(-self.data.imag))
        #Plot the amplitude and phase
        sim_amp = np.abs(s_solution)
        sim_phase = np.angle(s_solution)
        self.bode_amp.plot(self.data.freq, self.data.amplitude, linestyle = "None", marker = ".", markersize = 6, fillstyle = "full", color = "#2222A0", markeredgecolor = "#000000", markeredgewidth = 1)
        self.bode_amp.plot(sim_freq, sim_amp, linestyle = "-", marker = "None", linewidth = 1.5, color = "#4444DD")
        self.bode_phase.plot(self.data.freq, self.data.phase*180/np.pi, linestyle = "None", marker = "s", markersize = 6, fillstyle = "full", color = "#229950", markeredgecolor = "#000000", markeredgewidth = 1)
        self.bode_phase.plot(sim_freq, sim_phase*180/np.pi, linestyle = "-", marker = "None", linewidth = 1.5, color = "#44DD44")
        #Update the limits
        self.bode_amp.set_xlim(left = min(self.data.freq), right = max(self.data.freq))
        self.bode_amp.set_ylim(top = 1.3*max(self.data.amplitude), bottom = 0.7*min(self.data.amplitude))
        self.bode_phase.set_ylim(top = 1.1*max(self.data.phase)*180/np.pi, bottom = min(sim_phase)*180/np.pi - 0.1*180/np.pi*min(sim_phase)*min(sim_phase)/abs(min(sim_phase)))
        #Draw the refinement frequency limits
        if self.limit_visualisation:
            high, low = float(self.high_lim_input.get()), float(self.low_lim_input.get())
            self.residuals.plot([low, low], [-rlim, rlim], linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444", label = "Refinement limits")
            self.residuals.plot([high, high], [-rlim, rlim], linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444")
            self.bode_amp.plot([low, low], [0.7*min(self.data.amplitude), 1.3*max(self.data.amplitude)], linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444")
            self.bode_amp.plot([high, high], [0.7*min(self.data.amplitude), 1.3*max(self.data.amplitude)], linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444")
        #Add the legend
        self.residuals.legend(loc = "best", frameon = False, labelcolor = "linecolor", labelspacing = .05, handlelength = 1.25, handletextpad = .5)
        #Draw on the canvas
        self.canvas.draw()
    
    def display_error(self):
        """Calculate the reduced sum of the squares and display it as result_text in the result_label."""
        dof = 2*len(self.data.freq) - len(self.parameter_dict)
        redsumsq = (1/dof)*sum(((np.real(self.function_to_fit(self.refined_parameters, self.data.freq)) - self.data.real)/self.data.real)**2 + ((np.imag(self.function_to_fit(self.refined_parameters, self.data.freq)) - self.data.imag)/self.data.imag)**2)
        #total_error = sum(((np.abs(self.function_to_fit(self.refined_parameters, self.data.freq)) - self.data.amplitude)/self.data.amplitude)**2)
        self.result_text.set("S_v = {:g}".format(redsumsq))
    
    def refine_solution(self):
        """Indicate that refinement is in progress. Then apply boundaries, update the parameter history and minimize the error_function. Finally, update the parameters and residuals, and indicate that the refinement is finished."""
        #Indicate that refinement is in progress
        self.refine_text.set("Refining...")
        self.update()
        #Determine what to refine
        to_be_refined = []
        for t in self.tick_states:
            to_be_refined.append(t.get())
        #Refinement engine
        chosen_module = list(self.chosen_module_and_optimizer.get().split(":"))[0]
        chosen_method = list(self.chosen_module_and_optimizer.get().split(":"))[1].lstrip(" ")
        self.refinement_engine = RefinementEngine(self.function_to_fit, self.jnp_function_to_fit, self.data, self.refined_parameters, self.flipped_parameter_dict, to_be_refined, float(self.high_lim_value.get()), float(self.low_lim_value.get()), self.parameter_history, weighting_scheme = self.chosen_weighting.get(), opt_module = chosen_module, opt_method = chosen_method)
        #Refinement
        self.refinement_engine.refine_solution()
        self.refined_parameters = self.refinement_engine.parameters
        #Update
        self.set_param_labels() #Update parameter labels
        self.update_residuals() #Update residuals
        self.display_error() #Show total error
        self.refine_text.set("Refine solution")
    
    def previous_refinement(self):
        """Revert the parameters back to what they were before the most recent refinement."""
        self.refined_parameters = self.parameter_history[-1] #Go back one set of refined parameters
        self.set_param_labels() #Reset parameter labels
        self.update_residuals() #Reset residuals
        self.display_error() #Reset displayed error
    
    def reset_parameters(self):
        """Revert the parameters back to what they were when the window was launched."""
        self.refined_parameters = self.initial_parameters[:len(self.refined_parameters)] #Reset refined parameters
        self.set_param_labels() #Reset parameter labels
        self.update_residuals() #Reset residuals
        self.display_error() #Reset displayed error
        self.parameter_history = [] #Clear refinement history
        
    def accept_refinement(self):
        """Set accept_refinement to True and close the window. This will cause the model in DECiM core to be updated following the refinement result."""
        self.refined_parameters = list(np.concatenate((np.array(self.refined_parameters), np.zeros(100 - len(self.refined_parameters))), axis = None)) #Pad refinement output with zeros
        self.refinement_accepted = True #The refinement has been accepted
        self.destroy() #Close the window
        
    def reject_refinement(self):
        """Set accept_refinement to False and close the window. The model in DECiM core will remain unchanged."""
        self.refinement_accepted = False #The refinement has been rejected
        self.destroy() #Close the window
        
    def flip_parameter_values_keys(self):
        """Flip the keys and values of self.parameter_dict and store the new dict as self.flipped_parameter_dict."""
        self.flipped_parameter_dict = {}
        for p in self.parameter_dict:
            self.flipped_parameter_dict[self.parameter_dict[p]] = p
                    
    def toggle_all_parameters(self):
        """Disable or enable the refinement of ALL parameters."""
        on = 0
        off = 0
        for s in self.tick_states:
            if s.get() == 0:
                off += 1
            elif s.get() == 1:
                on += 1
        if on > off:
            for t in self.tick_boxes:
                t.deselect()
        else:
            for t in self.tick_boxes:
                t.select()
                
    def optimizer_tick_correction(self, *ev_args):
        """Disable tick boxes for parameter selection for incompatible modules/optimizers; enable them again for compatible ones.
        
        Arguments:
        self
        ev_args -- arguments given by the tk.StringVar trace"""
        if list(self.chosen_module_and_optimizer.get().split(":"))[0] == "Optax":
            for t in self.tick_boxes:
                t.configure(state = tk.DISABLED)
        elif list(self.chosen_module_and_optimizer.get().split(":"))[0] == "SciPy":
            for t in self.tick_boxes:
                t.configure(state = tk.NORMAL)