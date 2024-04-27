"""Part of DECiM. This file contains the interactive elements code. Last modified 27 April 2024 by Henrik Rodenburg.

Classes:
InteractionFrame -- contains all the controls for manual adjustment of fitting parameters"""

###########
##IMPORTS##
###########

import numpy as np

import tkinter as tk
import tkinter.ttk as ttk

from ecm_circuits import Circuit, Unit, Resistor, Capacitor, Inductor, ConstantPhaseElement, WarburgOpen, WarburgShort, HavriliakNegami, GerischerElement

###########################
##INTERACTIVE FRAME CLASS##
###########################

class InteractionFrame(ttk.Frame):
    def __init__(self, circuit, parameters, update_cmd):
        """"Contains dropdowns to select circuit elements and slider responses. Contains inputs for parameter min/max values.
        Also includes a slider, an input field that overrides it and a CLEAR button to clear said input field.
        
        Init arguments:
        circuit -- ecm_circuits.Circuit object being used by DECiM core
        parameters -- list of fit parameters
        update_cmd -- canvasUpdate method from DECiM core
        
        Attributes:
        circuit -- ecm_circuits.Circuit object being used by DECiM core
        parameters -- list of fit parameters
        
        chosen_parameter -- tk.StringVar representing the parameter being modified by the slider
        slider_scaling -- tk.StringVar representing the slider scaling mode (linear, logarithmic)
        lower_limit -- tk.StringVar representing the slider lower limit
        upper_limit -- tk.StringVar representing the slider upper limit
        parameter_value -- tk.StringVar holding the current parameter value
        override_parameter_value -- tk.StringVar holding the current value in the Entry widget
        
        parameter_frame, choice_label, parameter_dropdown -- frame, description label and tk.OptionMenu for choosing a parameter
        adjust_button -- button to automatically adjust slider_scaling, lower_limit and upper_limit based on parameter name
        response_frame, response_label, response_dropdown -- frame, description label and tk.OptionMenu for choosing the slider scaling mode
        error_label, error_frame, error_text -- label, frame and string for slider errors (controlled via DECiM core)
        low_lim_frame, low_lim_label, low_lim_box -- frame, description label and tk.Entry widget for choosing the lower frequency limit
        high_lim_frame, high_lim_label, high_lim_box -- frame, description label and tk.Entry widget for choosing the upper frequency limit
        slider_frame, slider, parameter_label -- frame, tk.Scale for setting the parameter value and a label to display the parameter name and value
        override_field_frame, override_label, override_box, override_button -- frame, description label, tk.Entry widget and button to override the slider input and instead type the desired parameter value
        
        Methods:
        update_cmd -- canvasUpdate method from DECiM core
        reset_parameter_dropdown -- clear the list of parameters in the dropdown and fill it again
        update_label -- update the parameter_value label
        parameter_index -- get the index in the fit parameter list of the chosen_parameter
        slider_set_parameter -- convert the slider value to a parameter value, then update the value of the parameter being modified
        override -- set the parameter value directly with the override button
        adjust_controls -- automatically adjust slider_scaling, lower_limit and upper_limit based on parameter name"""
        super().__init__() #Initialise the frame
        
        #Reference to the canvas update function
        self.update_cmd = update_cmd
        
        #Parameters are saved here
        self.circuit = circuit
        self.parameters = parameters
        
        #Variables
        self.chosen_parameter = tk.StringVar() #You can check which index this has in self.circuit_elements to determine the index of the parameter that is being modified.
        self.slider_scaling = tk.StringVar() #Determines how the slider value is converted to a parameter value.
        self.lower_limit = tk.StringVar() #Lower limit for the parameter value
        self.upper_limit = tk.StringVar() #Upper limit for the parameter value
        self.parameter_value = tk.StringVar() #Current value of the parameter, plus parameter name and units
        self.override_parameter_value = tk.StringVar() #Current override value of the parameter.
        
        #Create the different elements and pack them to the left
        
        #Parameter dropdown
        self.parameter_frame = ttk.Frame(self)
        self.choice_label = tk.Label(self.parameter_frame, text = "Parameter")
        self.choice_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.chosen_parameter.set("R0")
        self.parameter_dropdown = tk.OptionMenu(self.parameter_frame, self.chosen_parameter, "R0")
        self.parameter_dropdown.pack(side = tk.TOP, anchor = tk.CENTER)
        self.reset_parameter_dropdown()
        
        #Update button
        self.adjust_button = tk.Button(self.parameter_frame, text = "Adjust controls", command = self.adjust_controls)
        self.adjust_button.pack(side = tk.BOTTOM, anchor = tk.CENTER)
        self.parameter_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Response dropdown
        self.response_frame = ttk.Frame(self)
        self.response_label = tk.Label(self.response_frame, text = "Response")
        self.response_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.slider_scaling.set("logarithmic")
        self.response_dropdown = tk.OptionMenu(self.response_frame, self.slider_scaling, "logarithmic", "linear")
        self.response_dropdown.pack(side = tk.TOP, anchor = tk.CENTER)
        self.response_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Error label
        self.error_frame = ttk.Frame(self)
        self.error_label = tk.Label(self.response_frame, text = "Calculation error!", fg = "#c00", disabledforeground = "SystemButtonFace")
        self.error_label.configure(state = "disabled")
        self.error_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.error_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Slider lower limit input
        self.low_lim_frame = ttk.Frame(self)
        self.low_lim_label = tk.Label(self.low_lim_frame, text = "Limit 1")
        self.low_lim_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.lower_limit.set("0")
        self.low_lim_box = tk.Entry(self.low_lim_frame, textvariable = self.lower_limit)
        self.low_lim_box.pack(side = tk.BOTTOM, anchor = tk.CENTER)
        self.low_lim_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Slider
        self.slider_frame = ttk.Frame(self)
        self.slider = tk.Scale(self.slider_frame, from_ = 0, to = 10000, command = self.slider_set_parameter, length = 400, resolution = 1, orient = tk.HORIZONTAL)
        self.slider.to = 10000
        self.slider.pack(side = tk.TOP, anchor = tk.CENTER)
        
        #Label
        self.parameter_label = tk.Label(self.slider_frame, textvariable = self.parameter_value)
        self.parameter_label.pack(side = tk.BOTTOM, anchor = tk.CENTER)
        self.slider_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Slider upper limit input
        self.high_lim_frame = ttk.Frame(self)
        self.high_lim_label = tk.Label(self.high_lim_frame, text = "Limit 2")
        self.high_lim_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.upper_limit.set("1e12")
        self.high_lim_box = tk.Entry(self.high_lim_frame, textvariable = self.upper_limit)
        self.high_lim_box.pack(side = tk.BOTTOM, anchor = tk.CENTER)
        self.high_lim_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Override input
        self.override_field_frame = ttk.Frame(self)
        self.override_label = tk.Label(self.override_field_frame, text = "Direct input")
        self.override_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.override_parameter_value.set("0")
        self.override_box = tk.Entry(self.override_field_frame, textvariable = self.override_parameter_value)
        self.override_box.pack(side = tk.TOP, anchor = tk.CENTER)
        self.override_button = tk.Button(self.override_field_frame, text = "Set", command = self.override)
        self.override_button.pack(side = tk.BOTTOM, anchor = tk.CENTER)
        self.override_field_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
    
    def reset_parameter_dropdown(self):
        """Clears the parameter selection tk.OptionMenu and fills it again with new options based on the circuit diagram."""
        #From https://stackoverflow.com/questions/17580218/changing-the-options-of-a-optionmenu-when-clicking-a-button
        #Reset self.chosen_parameter and clear the dropdown
        self.chosen_parameter.set("")
        self.parameter_dropdown["menu"].delete(0, tk.END)
        #Couple the new circuit elements to self.chosen_parameter
        for element in self.circuit.diagram.list_elements():
            self.parameter_dropdown["menu"].add_command(label = element.name, command = tk._setit(self.chosen_parameter, element.name))
            if element.tag == "Q":
                self.parameter_dropdown["menu"].add_command(label = "n" + str(element.number), command = tk._setit(self.chosen_parameter, "n" + str(element.number)))
            if element.tag == "O":
                self.parameter_dropdown["menu"].add_command(label = "k" + str(element.number), command = tk._setit(self.chosen_parameter, "k" + str(element.number)))
            if element.tag == "S":
                self.parameter_dropdown["menu"].add_command(label = "l" + str(element.number), command = tk._setit(self.chosen_parameter, "l" + str(element.number)))
            if element.tag == "G":
                self.parameter_dropdown["menu"].add_command(label = "m" + str(element.number), command = tk._setit(self.chosen_parameter, "m" + str(element.number)))
            if element.tag == "H":
                self.parameter_dropdown["menu"].add_command(label = "t" + str(element.number), command = tk._setit(self.chosen_parameter, "t" + str(element.number)))
                self.parameter_dropdown["menu"].add_command(label = "b" + str(element.number), command = tk._setit(self.chosen_parameter, "b" + str(element.number)))
                self.parameter_dropdown["menu"].add_command(label = "g" + str(element.number), command = tk._setit(self.chosen_parameter, "g" + str(element.number)))
    
    def update_label(self, new_value):
        """Update the label below the slider with the (new) parameter name, value and units.
        
        Arguments:
        self
        new_value -- new value of the chosen parameter"""
        parameter_name = self.chosen_parameter.get()
        if parameter_name[0] in ["R", "O", "S", "G", "H"]:
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value) + " Ω")
        if parameter_name[0] == "C":
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value) + " F")
        if parameter_name[0] == "L":
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value) + " H")
        if parameter_name[0] == "Q":
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value) + " Fs^(n-1)")
        if parameter_name[0] in "nbg":
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value))
        if parameter_name[0] == "t":
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value) + "s^(b*g)")
        if parameter_name[0] in "klm":
            self.parameter_value.set(parameter_name + " = {:5g} ".format(new_value) + " s^(1/2)")
    
    def parameter_index(self):
        """Get the index in parameters of the currently chosen parameter."""
        parameter_name = self.chosen_parameter.get()
        for element in self.circuit.diagram.list_elements():
            if element.name == parameter_name: #R, C, L, Q, O, S, G: Parameter name matches element name. Can simply look for a match and return idx.
                return element.idx
            if element.tag == "Q" and parameter_name[0] == "n" and str(element.number) == parameter_name[1:]: #n: Find Q with the same number and return idx2
                return element.idx2
            if element.tag == "O" and parameter_name[0] == "k" and str(element.number) == parameter_name[1:]: #k: Find O with the same number and return idx2
                return element.idx2
            if element.tag == "S" and parameter_name[0] == "l" and str(element.number) == parameter_name[1:]: #l: Find S with the same number and return idx2
                return element.idx2
            if element.tag == "G" and parameter_name[0] == "m" and str(element.number) == parameter_name[1:]: #m: Find G with the same number and return idx2
                return element.idx2
            if element.tag == "H" and parameter_name[0] == "t" and str(element.number) == parameter_name[1:]: #t: Find H with the same number and return idx2
                return element.idx2
            if element.tag == "H" and parameter_name[0] == "b" and str(element.number) == parameter_name[1:]: #t: Find H with the same number and return idx3
                return element.idx3
            if element.tag == "H" and parameter_name[0] == "g" and str(element.number) == parameter_name[1:]: #t: Find H with the same number and return idx4
                return element.idx4
    
    def slider_set_parameter(self, event):
        """Convert the slider value to a parameter value, then update the parameter value and the parameter value label."""
        #Determine slider response
        low = float(self.lower_limit.get())
        high = float(self.upper_limit.get())
        scale = self.slider_scaling.get()
        #Determine output value
        if scale == "linear":
            #Lowest value is starting point (0). The relative value of the slider is determined through division by its maximum (self.slider.to) and multiplied with the remainder (high-low)
            output_value = low + float(self.slider.get())*(high - low)/self.slider.to
        if scale == "logarithmic":
            #This handles positive and negative log alike.
            output_value = low + 10**(float(self.slider.get())*np.log10(high)/self.slider.to)
        #Update the label
        self.update_label(output_value)
        #Update the parameter
        self.parameters[self.parameter_index()] = output_value
        #Update the canvas
        self.update_cmd()
    
    def override(self):
        """When pressing the 'set' button, set the parameter value and label."""
        #Get the output value
        output_value = float(self.override_parameter_value.get())
        #Update the label
        self.update_label(output_value)
        #Update the parameter
        self.parameters[self.parameter_index()] = output_value
        #Update the canvas
        self.update_cmd()
        self.adjust_controls()
        
    def adjust_controls(self):
        """Upon pressing the 'adjust controls' button, checks the parameter name and applies logical upper and lower limits for the slider, as well as logical slider scaling."""
        #Check what kind of parameter we're dealing with
        parameter_name = self.chosen_parameter.get()
        #Set the controls to logical values for the parameter type
        if parameter_name[0] in ["R", "O", "S", "G", "H"]:
            self.lower_limit.set("0")
            self.upper_limit.set("1e10")
            self.slider_scaling.set("logarithmic")
        if parameter_name[0] in ["C", "Q", "L", "m", "t"]:
            self.lower_limit.set("0")
            self.upper_limit.set("1e-12")
            self.slider_scaling.set("logarithmic")
        if parameter_name[0] in ["n", "b", "g"]:
            self.lower_limit.set("0")
            self.upper_limit.set("1")
            self.slider_scaling.set("linear")
        if parameter_name[0] in ["k", "l"]:
            self.lower_limit.set("0")
            self.upper_limit.set("1e6")
            self.slider_scaling.set("logarithmic")
        #Get the parameter value
        parameter_value = self.parameters[self.parameter_index()]
        #Update the direct input field
        self.override_parameter_value.set("{:5g}".format(parameter_value))
        #Get the slider response
        #Determine slider response
        low = float(self.lower_limit.get())
        high = float(self.upper_limit.get())
        scale = self.slider_scaling.get()
        #Set the slider to the expected value
        if scale == "logarithmic":
            self.slider.set(int(self.slider.to*np.log10(parameter_value - low)/np.log10(high - low)))
        if scale == "linear":
            self.slider.set(int(self.slider.to*(parameter_value - low)/(high - low)))