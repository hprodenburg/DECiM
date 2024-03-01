"""Part of DECiM. This file contains the fitting classes. Last modified 8 December 2023 by Henrik Rodenburg.

Classes:
RefinementEngine -- class for the simple refinement
RefinementWindow -- class for the advanced refinement"""

###########
##IMPORTS##
###########

import numpy as np
import scipy.optimize as op

import matplotlib as mp
import matplotlib.pyplot as pt
import matplotlib.backends.backend_tkagg as btk
import matplotlib.figure as fg
import matplotlib.animation as anim

import tkinter as tk
import tkinter.ttk as ttk

from ecm_datastructure import dataSet
from ecm_helpers import nearest

#####################
##SIMPLE REFINEMENT##
#####################

class RefinementEngine():
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
        self.data = data #Raw data, taking the form of a dataSet object (ecm_datastructure). If a limited frequency range is desired, the truncated data should be passed to the RefinementEngine.
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

#######################
##ADVANCED REFINEMENT##
#######################

class RefinementWindow(tk.Toplevel):
    def __init__(self, function_to_fit, element_list, initial_parameters, data):
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
        reject_refinement -- close the window and do not change the model parameters"""
        super().__init__()
        self.title("Equivalent circuit model refinement")
        self.width = int(self.winfo_screenwidth()*0.85)
        self.height = int(self.winfo_screenheight()*0.85)
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        
        #Variables
        self.function_to_fit = function_to_fit
        self.element_list = element_list
        self.parameter_dict = {}
        indices = []
        for p in self.element_list: #Tie parameter names to parameter array indices and count the parameters
            self.parameter_dict[p.name] = p.idx
            indices.append(p.idx)
            if p.tag in "QSOG":
                indices.append(p.idx2)
                if p.tag == "Q":
                    self.parameter_dict["n" + str(p.number)] = p.idx2
                if p.tag == "O":
                    self.parameter_dict["k" + str(p.number)] = p.idx2
                if p.tag == "S":
                    self.parameter_dict["l" + str(p.number)] = p.idx2
                if p.tag == "G":
                    self.parameter_dict["m" + str(p.number)] = p.idx2
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
        self.tick_label = tk.Label(self.tick_frame, text = "Parameters to refine")
        self.tick_label.pack(side = tk.TOP, anchor = tk.W)
        self.tick_box_frame = ttk.Frame(self.tick_frame) #The frame is subdivided into two colums. One for boxes...
        self.tick_box_frame.pack(side = tk.LEFT, anchor = tk.W)
        self.tick_label_frame = ttk.Frame(self.tick_frame) #... and one for labels.
        self.tick_label_frame.pack(side = tk.LEFT, anchor = tk.W)
        for param in self.parameter_list: #A tick box and label are made for every parameter.
            self.tick_states.append(tk.IntVar(self))
            self.tick_boxes.append(tk.Checkbutton(self.tick_box_frame, text = param, variable = self.tick_states[-1]))
            self.parameter_values.append(tk.StringVar(self))
            self.parameter_labels.append(tk.Label(self.tick_label_frame, textvariable = self.parameter_values[-1]))
            if str(param)[0] in "RLCQnOSGklm": #Only pack real, meaningful parameters.
                self.tick_boxes[-1].pack(side = tk.TOP, anchor = tk.W, padx = 30)
                self.parameter_labels[-1].pack(side = tk.TOP, anchor = tk.E)
        self.set_param_labels()
        
    def make_limit_frame(self):
        """Create tk.Entries and tk.Labels for setting the refinement frequency limits, as well as a tk.OptionMenu from which the weighting schemes can be chosen."""
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
            units = "" #Stays this way for n
            if str(p)[0] in ["R", "O", "S", "G"]:
                units = " Ohm"
            if str(p)[0] == "C":
                units = " F"
            if str(p)[0] == "Q":
                units = " Fs^(n-1)"
            if str(p)[0] == "L":
                units = " H"
            if str(p)[0] in ["k", "l", "m"]:
                units = " s^(1/2)"
            if str(p)[0] in "RLCQnOSGklm":
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
    
    def error_function(self, params):
        """Function to be minimized in the refinement. Weighting schemes are applied here.
        
        Arguments:
        self
        params -- list of fit parameters
        
        Returns:
        NumPy array; sum of the squared differences in the impedance; this is done separately for the real and imaginary components, which are then added."""
        w_re, w_im = 1, 1 #Unit weighting is default
        wsch = self.chosen_weighting.get()
        if wsch == "Calculated modulus":
            w_re = np.sqrt(np.real(self.function_to_fit(params, self.data.freq))**2 + np.imag(self.function_to_fit(params, self.data.freq))**2)
            w_im = w_re
        elif wsch == "Observed modulus":
            w_re = np.sqrt(self.data.real**2 + self.data.imag**2)
            w_im = w_re
        elif wsch == "Calculated proportional":
            w_re = 1/(np.real(self.function_to_fit(params, self.data.freq))**2)
            w_im = 1/(np.imag(self.function_to_fit(params, self.data.freq))**2)
        elif wsch == "Observed proportional":
            w_re = 1/(self.data.real**2)
            w_im = 1/(self.data.imag**2)
        if self.l_idx == 0 and self.h_idx == -1:
            return sum(w_re*(np.real(self.function_to_fit(params, self.data.freq)) - self.data.real)**2) + sum(w_im*(np.imag(self.function_to_fit(params, self.data.freq)) - self.data.imag)**2)
        #If limits are set, return the limited result (with weights being taken care of first)
        if wsch not in ["", "Unit"]:
            w_re, w_im = w_re[self.l_idx:self.h_idx], w_im[self.l_idx:self.h_idx]
        return sum(w_re*(np.real(self.function_to_fit(params, self.data.freq[self.l_idx:self.h_idx])) - self.data.real[self.l_idx:self.h_idx])**2) + sum(w_im*(np.imag(self.function_to_fit(params, self.data.freq[self.l_idx:self.h_idx])) - self.data.imag[self.l_idx:self.h_idx])**2)
    
    def display_error(self):
        """Calculate the reduced sum of the squares and display it as result_text in the result_label."""
        dof = 2*len(self.data.freq) - len(self.parameter_dict)
        redsumsq = (1/dof)*sum(((np.real(self.function_to_fit(self.refined_parameters, self.data.freq)) - self.data.real)/self.data.real)**2 + ((np.imag(self.function_to_fit(self.refined_parameters, self.data.freq)) - self.data.imag)/self.data.imag)**2)
        #total_error = sum(((np.abs(self.function_to_fit(self.refined_parameters, self.data.freq)) - self.data.amplitude)/self.data.amplitude)**2)
        self.result_text.set("S_v = {:g}".format(redsumsq))
        
    def optimisation_callback(self, params):
        """Print the output of the refinement function to the command line for each iteration."""
        if self.silent_optimisation:
            return
        out_str = "Iteration " + str(self.iter) + " | "
        for p in self.parameter_dict:
            out_str += p + ": " + "{:5g} | ".format(params[self.parameter_dict[p]])
        print(out_str)
        self.iter += 1
    
    def refine_solution(self):
        """Indicate that refinement is in progress. Then apply boundaries, update the parameter history and minimize the error_function. Finally, update the parameters and residuals, and indicate that the refinement is finished."""
        #Indicate that refinement is in progress
        self.refine_text.set("Refining...")
        self.update()
        #Frequency limits
        high, low = float(self.high_lim_input.get()), float(self.low_lim_input.get())
        self.h_idx, self.l_idx = nearest(high, self.data.freq), nearest(low, self.data.freq)
        #Boundary determination
        boundaries = []
        for t in range(len(self.tick_states)):
            if self.tick_states[t].get() == 0: #Parameter not included in refinement
                boundaries.append((self.refined_parameters[t], self.refined_parameters[t])) #Constrain parameter to its present value
            elif self.tick_states[t].get() == 1: #Parameter included in refinement
                if self.parameter_list[t][0] == "n":
                    boundaries.append((0, 1)) #Refine the parameter (n bounded between 0 and 1)
                else:  
                    boundaries.append((0, 1e15)) #Refine the parameter (other parameters bounded between 0 and 1e15)
            else:
                print("Warning: forbidden tick state. Constraining parameter {:s}.".format(self.parameter_list[t]))
                boundaries.append((self.refined_parameters[t], self.refined_parameters[t]))
        #Save previous refinement
        self.parameter_history.append(self.refined_parameters)
        #Refinement
        self.iter = 0
        opt_res = op.minimize(self.error_function, np.array(self.refined_parameters), method="Nelder-Mead", options = {"maxiter": 500*len(self.refined_parameters)}, bounds = boundaries, callback = self.optimisation_callback) #Minimise the error
        self.refined_parameters = opt_res["x"] #Save new parameters
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
        