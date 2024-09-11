"""Part of DECiM. This file contains the interactive elements code. Last modified 11 September 2024 by Henrik Rodenburg.

Classes:
InteractionFrame -- contains all the controls for manual adjustment of fitting parameters
ScrollableListbox -- combination of tk.Listbox and tk.Scrollbar, used for selecting parameters"""

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
        listbox_indices -- list of parameter indices in self.parameter_listbox
        
        chosen_parameter -- tk.StringVar representing the parameter being modified by the slider
        slider_scaling -- tk.StringVar representing the slider scaling mode (linear, logarithmic)
        lower_limit -- tk.StringVar representing the slider lower limit
        upper_limit -- tk.StringVar representing the slider upper limit
        parameter_value -- tk.StringVar holding the current parameter value
        override_parameter_value -- tk.StringVar holding the current value in the Entry widget
        
        parameter_frame, choice_label, parameter_listbox -- frame, description label and ScrollableListbox for choosing a parameter
        adjust_button -- button to automatically adjust slider_scaling, lower_limit and upper_limit based on parameter name
        response_frame, response_label, response_dropdown -- frame, description label and tk.OptionMenu for choosing the slider scaling mode
        error_label, error_frame, error_text -- label, frame and string for slider errors (controlled via DECiM core)
        low_lim_frame, low_lim_label, low_lim_box -- frame, description label and tk.Entry widget for choosing the lower frequency limit
        high_lim_frame, high_lim_label, high_lim_box -- frame, description label and tk.Entry widget for choosing the upper frequency limit
        slider_frame, slider, parameter_label -- frame, tk.Scale for setting the parameter value and a label to display the parameter name and value
        override_field_frame, override_label, override_box, override_button -- frame, description label, tk.Entry widget and button to override the slider input and instead type the desired parameter value
        
        as_refined -- Boolean; indicates if result is as refined or not; partly controlled via DECiM core
        
        Methods:
        update_cmd -- canvasUpdate method from DECiM core
        reset_parameter_listbox -- clear the list of parameters in the ScrollableListbox and fill it again
        select_parameter -- retrieve the currently selected parameter in self.parameter_listbox (when the user clicks)
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
        
        #Refinement check parameter
        self.as_refined = False
        
        #Variables
        self.chosen_parameter = tk.StringVar() #You can check which index this has in self.circuit_elements to determine the index of the parameter that is being modified.
        self.slider_scaling = tk.StringVar() #Determines how the slider value is converted to a parameter value.
        self.lower_limit = tk.StringVar() #Lower limit for the parameter value
        self.upper_limit = tk.StringVar() #Upper limit for the parameter value
        self.parameter_value = tk.StringVar() #Current value of the parameter, plus parameter name and units
        self.override_parameter_value = tk.StringVar() #Current override value of the parameter.
        
        #Create the different elements and pack them to the left
        
        #Parameter ScrollableListbox -- need to change self.chosen_parameter to have information on parameter value
        self.parameter_frame = ttk.Frame(self)
        self.choice_label = tk.Label(self.parameter_frame, text = "Parameter")
        self.choice_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.chosen_parameter.set("R0: 1")
        self.parameter_listbox = ScrollableListbox(self.parameter_frame)
        self.parameter_listbox.bind_select(self.select_parameter)
        self.parameter_listbox.pack(side = tk.TOP, anchor = tk.CENTER)
        self.listbox_indices = {}
        self.reset_parameter_listbox()
        
        #Update button
        self.response_frame = ttk.Frame(self)
        self.adjust_button = tk.Button(self.response_frame, text = "Adjust controls", command = self.adjust_controls)
        self.adjust_button.pack(side = tk.BOTTOM, anchor = tk.CENTER)
        self.parameter_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.BOTH, expand = True)
        
        #Response dropdown
        self.response_label = tk.Label(self.response_frame, text = "Response")
        self.response_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.slider_scaling.set("logarithmic")
        self.response_dropdown = tk.OptionMenu(self.response_frame, self.slider_scaling, "logarithmic", "linear")
        self.response_dropdown.pack(side = tk.TOP, anchor = tk.CENTER)
        self.response_frame.pack(side = tk.LEFT, anchor = tk.E, fill = tk.X, expand = True)
        
        #Error label
        self.error_frame = ttk.Frame(self)
        self.error_label = tk.Label(self.response_frame, text = "Calculation error!", fg = "#c00", disabledforeground = "SlateGray1")
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
        self.slider.configure(state = tk.DISABLED)
        
        #Label
        self.parameter_label = tk.Label(self.slider_frame, textvariable = self.parameter_value)
        self.parameter_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.parameter_value.set("No parameter selected")
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
                
    def reset_parameter_listbox(self):
        """Clears the parameter selection ScrollableListbox and fills it again with new options based on the circuit diagram.
        
        Arguments:
        self"""
        self.parameter_listbox.clear()
        i = 0
        #Couple the new circuit elements to self.chosen_parameter
        for element in self.circuit.diagram.list_elements():
            self.parameter_listbox.insert(element.name + ": {:5g}".format(self.parameters[self.parameter_index(element.name)]))
            self.listbox_indices[element.name] = i
            i += 1
            if element.tag == "Q":
                self.parameter_listbox.insert("n" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("n" + str(element.number))]))
                self.listbox_indices["n" + str(element.number)] = i
                i += 1
            if element.tag == "O":
                self.parameter_listbox.insert("k" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("k" + str(element.number))]))
                self.listbox_indices["k" + str(element.number)] = i
                i += 1
            if element.tag == "S":
                self.parameter_listbox.insert("l" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("l" + str(element.number))]))
                self.listbox_indices["l" + str(element.number)] = i
                i += 1
            if element.tag == "G":
                self.parameter_listbox.insert("m" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("m" + str(element.number))]))
                self.listbox_indices["m" + str(element.number)] = i
                i += 1
            if element.tag == "H":
                self.parameter_listbox.insert("t" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("t" + str(element.number))]))
                self.listbox_indices["t" + str(element.number)] = i
                i += 1
                self.parameter_listbox.insert("b" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("b" + str(element.number))]))
                self.listbox_indices["b" + str(element.number)] = i
                i += 1
                self.parameter_listbox.insert("g" + str(element.number) + ": {:5g}".format(self.parameters[self.parameter_index("g" + str(element.number))]))
                self.listbox_indices["g" + str(element.number)] = i
                i += 1
                
    def select_parameter(self, event):
        """Select a parameter from self.parameter_listbox. Disable slider if no parameter is selected.
        
        Arguments:
        self
        event -- The user's action of clicking on the listbox."""
        if len(self.parameter_listbox.get())> 0:
            if self.parameter_listbox.get()[0] in "RLCQOSGHklmntbg":
                self.slider.configure(state = tk.NORMAL)
                self.chosen_parameter.set(self.parameter_listbox.get())
                self.reset_parameter_listbox()
                self.parameter_listbox.itemconfig(self.listbox_indices[list(self.chosen_parameter.get().split(":"))[0]], {"bg": "#00f", "fg": "#fff"})
        else:
            self.slider.configure(state = tk.DISABLED)
            self.parameter_value.set("No parameter selected")
            
    def update_label(self, new_value):
        """Update the label below the slider with the (new) parameter name, value and units.
        
        Arguments:
        self
        new_value -- new value of the chosen parameter"""
        parameter_name = list(self.chosen_parameter.get().split(":"))[0]
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
    
    def parameter_index(self, parameter_name):
        """Get the index in parameters of the currently chosen parameter."""
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
        self.parameters[self.parameter_index(list(self.chosen_parameter.get().split(":"))[0])] = output_value
        #Clear the as-refined status
        self.as_refined = False
        #Update the canvas
        self.update_cmd()
    
    def override(self):
        """When pressing the 'set' button, set the parameter value and label."""
        #Get the output value
        output_value = float(self.override_parameter_value.get())
        #Update the label
        self.update_label(output_value)
        #Update the parameter
        self.parameters[self.parameter_index(list(self.chosen_parameter.get().split(":"))[0])] = output_value
        #Clear the as-refined status
        self.as_refined = False
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
        parameter_value = self.parameters[self.parameter_index(list(self.chosen_parameter.get().split(":"))[0])]
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
            
class ScrollableListbox():
    def __init__(self, master, select_mode = tk.SINGLE):
        """Listbox-scrollbar combination.
        
        Init arguments:
        self
        master -- ttk.Frame or tk.Frame on which to place the ScrollableListbox
        select_mode -- selectmode for self.listbox; tk.SINGLE for DECiM Core, tk.MULTIPLE for Advanced Refinement
        
        Attributes:
        listbox -- tk.Listbox
        scrollbar -- tk.Scrollbar
        
        Methods:
        insert -- insert value into self.listbox
        get -- get the currently selected option in self.listbox (tk.SINGLE selection mode)
        get_all -- get all currently selected options in self.listbox (tk.MULTIPLE selection mode)
        pack -- pack self.listbox and self.scrollbar and configure the two to work together
        bind_select -- bind a function to element selection
        itemconfig -- self.listbox.itemconfig"""
        self.listbox = tk.Listbox(master, activestyle = tk.NONE, selectmode = select_mode)
        self.scrollbar = tk.Scrollbar(master)
        self.itemconfig = self.listbox.itemconfig
        
    def insert(self, value):
        """Put a value (string) into self.listbox.
        
        Arguments:
        self
        value -- String, to be added to listbox"""
        self.listbox.insert(tk.END, value)
        
    def get(self):
        """Get the selected value in self.listbox in tk.SINGLE selection mode.
        
        Arguments:
        self
        
        Returns:
        Currently selected value in self.listbox."""
        return self.listbox.get(tk.ANCHOR)
        
    def get_all(self):
        """Get all selected values in self.listbox in tk.MULTIPLE selection mode.
        
        Arguments:
        self
        
        Returns:
        List of all selected values in self.listbox"""
        outpars = []
        for i in self.listbox.curselection():
            outpars.append(self.listbox.get(i))
        return outpars
        
    def clear(self):
        """Delete all values in self.listbox."""
        self.listbox.delete(0, tk.END)
        
    def bind_select(self, function):
        """Bind a function to self.listbox("<<ListboxSelect>>")
        
        Arguments:
        self
        function -- Function to bind"""
        self.listbox.bind("<<ListboxSelect>>", function)
        
    def pack(self, side = tk.LEFT, fill = tk.NONE, anchor = tk.W, expand = tk.FALSE):
        """Pack self on master and configure self.scrollbar to scroll through self.listbox.
        
        Arguments:
        self
        side -- side on which to pack self
        fill -- which directions to fill on master
        anchor -- packing anchor
        expand -- expansion setting (horizontal, vertical, both)"""
        #Attach listbox and scrollbar to each other
        self.listbox.config(yscrollcommand = self.scrollbar.set)
        self.scrollbar.config(command = self.listbox.yview)
        #Packing -- listbox only, scrollbar is invisible, but the scroll wheel works
        self.listbox.pack(side = side, fill = fill, anchor = anchor)
        