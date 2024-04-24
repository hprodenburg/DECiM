"""Part of DECiM. This file contains circuit arrangement code, impedance calculations and circuit selection UI code. Last modified 24 April 2024 by Henrik Rodenburg.

Classes:
Circuit elements:
Resistor
Capacitor
Inductor
ConstantPhaseElement
WarburgOpen
WarburgShort
HavriliakNegami
GerischerElement

Circuit construction:
Unit
Circuit

Interaction with DECiM core:
CircuitManager

UI:
PannableCanvas
CircuitDefinitionWindow"""

###########
##IMPORTS##
###########

import numpy as np
import copy

import tkinter as tk

import ecm_custom_models as ecmcm

######################
##CIRCUIT COMPONENTS##
######################

class Resistor():
    def __init__(self, resistance, number, idx, parent_unit):
        """Resistor circuit element.
        
        Init arguments:
        resistance -- resistance
        number -- number of this element within its specific element type
        idx -- number of this element among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        tag -- identifies the type of element
        R -- resistance
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element
        """
        self.tag = "R"
        self.R = resistance
        self.number = number #Element number. This gives the name of the element when combined with its tag.
        self.idx = idx #Element index. This determines which index in the parameter list corresponds to the element.
        self.name = self.tag + str(self.number)
        self.parent_units = [] #List of units that this element is part of, from lowest to highest order.
        self.parent_units.append(parent_unit)
    
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        value of resistance"""
        return self.R

class Capacitor():
    def __init__(self, capacitance, number, idx, parent_unit):
        """Capacitor circuit element.
        
        Init arguments:
        capacitance -- capacitance
        number -- number of this element within its specific element type
        idx -- number of this element among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        tag -- identifies the type of element
        C -- capacitance
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element
        """
        self.tag = "C"
        self.C = capacitance
        self.number = number
        self.idx = idx
        self.name = self.tag + str(self.number)
        self.parent_units = []
        self.parent_units.append(parent_unit)
        
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        Complex NumPy array of impedances"""
        return 1/(1j*frequency*2*np.pi*self.C)
        
class Inductor():
    def __init__(self, inductance, number, idx, parent_unit):
        """Inductor circuit element.
        
        Init arguments:
        inductance -- inductance
        number -- number of this element within its specific element type
        idx -- number of this element among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        tag -- identifies the type of element
        L -- inductance
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element
        """
        self.tag = "L"
        self.L = inductance
        self.number = number
        self.idx = idx
        self.name = self.tag + str(self.number)
        self.parent_units = []
        self.parent_units.append(parent_unit)
        
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        Complex NumPy array of impedances"""
        return 1j*self.L*2*np.pi*frequency

class ConstantPhaseElement():
    def __init__(self, base, exponent, number, idx, parent_unit):
        """ConstantPhaseElement (CPE) circuit element.
        
        Init arguments:
        base -- magnitude; equivalent to C if n == 1 or R if n == 0
        exponent -- CPE exponent
        number -- number of this element within its specific element type
        idx -- number of this element's first parameter among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        tag -- identifies the type of element
        Q -- magnitude; equivalent to C if n == 1 or R if n == 0
        n -- exponent
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element's first parameter among all parameters
        idx2 -- number of this element's second parameter among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element
        """
        self.tag = "Q"
        self.Q = base
        self.n = exponent
        self.number = number
        self.idx = idx
        self.idx2 = idx + 1 #The CPE, Q, (and elements S, O, G) have two parameters and therefore need a second index.
        self.name = self.tag + str(self.number)
        self.parent_units = []
        self.parent_units.append(parent_unit)
        
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        Complex NumPy array of impedances"""
        return 1/(self.Q*(1j*frequency*2*np.pi)**self.n)
        
class WarburgOpen():
    def __init__(self, magnitude, k, number, idx, parent_unit):
        """WarburgOpen circuit element.
        
        Init arguments:
        magnitude -- magnitude
        k -- second parameter
        number -- number of this element within its specific element type
        idx -- number of this element's first parameter among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        tag -- identifies the type of element
        O -- magnitude
        k -- second parameter
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element's first parameter among all parameters
        idx2 -- number of this element's second parameter among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element
        """
        self.tag = "O"
        self.O = magnitude
        self.k = k
        self.number = number
        self.idx = idx
        self.idx2 = idx + 1
        self.name = self.tag + str(self.number)
        self.parent_units = []
        self.parent_units.append(parent_unit)
        
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        Complex NumPy array of impedances"""
        return self.O * (1j*frequency*self.k*2*np.pi)**-0.5 * (1/np.tanh((1j*frequency*self.k)**0.5))
        
class WarburgShort():
    def __init__(self, magnitude, l, number, idx, parent_unit):
        """WarburgShort circuit element.
        
        Init arguments:
        magnitude -- magnitude
        l -- second parameter
        number -- number of this element within its specific element type
        idx -- number of this element's first parameter among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        tag -- identifies the type of element
        S -- magnitude
        l -- second parameter
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element's first parameter among all parameters
        idx2 -- number of this element's second parameter among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element
        """
        self.tag = "S"
        self.S = magnitude
        self.l = l
        self.number = number
        self.idx = idx
        self.idx2 = idx + 1
        self.name = self.tag + str(self.number)
        self.parent_units = []
        self.parent_units.append(parent_unit)
    
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        Complex NumPy array of impedances"""
        return self.S * (1j*frequency*self.l*2*np.pi)**-0.5 * np.tanh((1j*frequency*self.l)**0.5)

class HavriliakNegami():
    def __init__(self, magnitude, m, beta, gamma, number, idx, parent_unit):
        """HavriliakNegami circuit element.
        
        Init arguments:
        magnitude -- magnitude
        m -- timescale
        beta -- exponent for j*omega*tau
        gamma -- exponent for denominator
        
        Attributes:
        tag -- identifies the type of element
        G -- magnitude
        m -- timescale
        number -- number of this element within its specific element type
        name -- string combining tag and number
        idx -- number of this element's first parameter (magnitude) among all parameters
        idx2 -- number of this element's second parameter (timescale) among all parameters
        idx3 -- number of this element's third parameter (beta) among all parameters
        idx4 -- number of this element's fourth parameter (gamma) among all parameters
        parent_units -- list of Units that contain this object; parent_units[0] is the lowest level Unit that contains it
        
        Methods:
        Z -- returns the impedance for this element"""
        self.tag = "H"
        self.G = magnitude
        self.m = m
        self.beta = beta
        self.gamma = gamma
        self.number = number
        self.idx = idx
        self.idx2 = idx + 1
        self.idx3 = idx + 2
        self.idx4 = idx + 3
        self.name = self.tag + str(self.number)
        self.parent_units = []
        self.parent_units.append(parent_unit)
        
    def Z(self, frequency):
        """Impedance function.
        
        Arguments:
        self -- circuit element
        frequency -- NumPy array of frequencies
        
        Returns:
        Complex NumPy array of impedances"""
        return self.G/((1 + (1j*frequency*self.m)**self.beta)**self.gamma)
        
class GerischerElement(HavriliakNegami):
    def __init__(self, magnitude, m, number, idx, parent_unit):
        """GerischerElement circuit element. Special case of HavriliakNegami.
        
        Init arguments:
        magnitude -- magnitude
        m -- second parameter
        number -- number of this element within its specific element type
        idx -- number of this element's first parameter among all parameters
        parent_unit -- the lowest level Unit that contains this element
        
        Attributes:
        See HavriliakNegami class.
        
        Methods:
        Z -- returns the impedance for this element (inherited from HavriliakNegami)"""
        super().__init__(magnitude, m, 1, 0.5, number, idx, parent_unit)
        self.tag = "G"

class Unit():
    def __init__(self, number, arrangement = "series"):
        """Unit circuit component. Essentially a circuit element that bundles other circuit elements. Units can contain both normal circuit elements and other Units.
        
        Init arguments:
        number -- Unit number
        arrangement -- string: series or parallel
        
        Attributes:
        tag -- string containing only U
        elements -- list of elements contained within the Unit
        tags -- list of tags of all elements contained within the Unit
        arrangement -- string to indicate if the elements in the unit are connected in series or in parallel
        number -- Unit number
        name -- string, combination of tag and number
        parent_units -- list of Units containing this unit; lowest level first
        
        Methods:
        add_element -- add an element to the unit
        list_elements -- list all elements, except Units; instead, Units contained within this parent Unit will list their own elements
        generate_circuit_string -- create the circuit string which describes this Unit
        ZSER -- impedance function for series Units
        ZPAR -- impedance function for parallel Units
        Z -- general impedance function"""
        self.tag = "U"
        self.elements = []
        self.tags = []
        self.arrangement = arrangement #String, 'series' or 'parallel'
        self.number = number
        self.name = self.tag + str(number)
        self.parent_units = []
    
    def add_element(self, element):
        """Add an element to the Unit; it is added to the list of elements and its tag is added to the list of tags.
        
        Arguments:
        self -- need access to Unit
        element -- circuit element to be added"""
        self.elements.append(element)
        self.tags.append(element.tag)
    
    #Provide list of elements and elements of subunits
    def list_elements(self):
        """Provide a list of all circuit elements. When encountering a unit, call its list_elements method.
        
        Arguments:
        self
        
        Returns:
        list containing all elements, including those in subunits; units themselves are not listed"""
        outlist = []
        for e in self.elements:
            if e.tag == "U":
                sublist = e.list_elements()
                outlist = outlist + sublist
            else:
                outlist.append(e)
        return outlist
        
    #Find a given unit within another unit based on its number
    def find_unit(self, number):
        """Find the unit U whose U.number == number.
        
        Arguments:
        self
        number -- integer, number of desired unit
        
        Returns:
        Unit object whose number attribute matches the number argument."""
        if self.number == number:
            return self
        else:
            for e in self.elements:
                if e.tag == "U":
                    e.find_unit(number)
        
    def generate_circuit_string(self, verbose = True):
        """Generate the circuit string of the Unit. Used in RECM2 files and in building the menus in DECiM core
        
        Arguments:
        self
        verbose -- Boolean; if True, add element numbers to the circuit string.
        
        Returns:
        Circuit string"""
        arr_dict = {"parallel": ("(", ")"),  "series": ("{", "}")}
        outstr = "" + arr_dict[self.arrangement][0]
        for e in self.elements:
            if e.tag == "U":
                outstr += e.generate_circuit_string()
            else:
                outstr += e.tag
                if verbose:
                    outstr += str(e.number)
        outstr += arr_dict[self.arrangement][1]
        return outstr
        
    def ZSER(self, elements, frequency):
        """Series impedance function.
        
        Arguments:
        elements -- list of elements
        frequency -- NumPy array of frequencies
        
        Returns:
        NumPy array of complex impedances"""
        sres = 0
        for elem in elements:
            sres += elem.Z(frequency)
        return sres
        
    #Parallel circuit function
    def ZPAR(self, elements, frequency):
        """Parallel impedance function.
        
        Arguments:
        elements -- list of elements
        frequency -- NumPy array of frequencies
        
        Returns:
        NumPy array of complex impedances"""
        sres = 0
        for elem in elements:
            sres += 1/elem.Z(frequency)
        return 1/sres
        
    #Impedance function for the unit
    def Z(self, frequency):
        """General impedance function for a Unit.
        
        Arguments:
        frequency -- NumPy array of frequencies
        
        Returns:
        The Unit's impedance: a NumPy array of complex impedances"""
        if self.arrangement == "parallel" and len(self.list_elements()) > 0:
            return self.ZPAR(self.elements, frequency)
        if self.arrangement == "series":
            return self.ZSER(self.elements, frequency)

#################
##CIRCUIT CLASS##
#################

#A circuit must always be defined while DECiM is running.
class Circuit():
    def __init__(self, diagram = Unit(0, arrangement = "parallel")):
        """Complete circuit.
        
        Init keyword arguments:
        diagram -- Unit describing the complete circuit
        
        Attributes:
        diagram -- Unit describing the complete circuit
        
        Methods:
        index_cleanup -- make sure the lowest element index is 0
        set_element_values -- match elements' values to those of the fit parameters
        impedance -- compute the circuit's impedance"""
        self.diagram = diagram
        self.index_cleanup()
        
    def index_cleanup(self):
        """Clean up the indices of the elements' parameters; make sure the lowest one is 0.
        
        Arguments:
        self"""
        min_index = 99 #The maximum parameter index is 99 for np.zeros(100) as the default parameter array
        for element in self.diagram.list_elements(): #First determine the lowest index
            if element.idx < min_index:
                min_index = element.idx
        for element in self.diagram.list_elements(): #Then subtract the minimum index
            element.idx -= min_index
            if element.tag in "QOSGH":
                element.idx2 -= min_index
            if element.tag == "H":
                element.idx3 -= min_index
                element.idx4 -= min_index
        
    def set_element_values(self, fitparams):
        """Assign all elements' parameters their values from the fit parameters list. This must be done before the impedance can be computed.
        
        Arguments:
        self
        fitparams -- list of fit parameters"""
        for element in self.diagram.list_elements():
            if element.tag == "R":
                element.R = fitparams[element.idx]
            if element.tag == "C":
                element.C = fitparams[element.idx]
            if element.tag == "L":
                element.L = fitparams[element.idx]
            if element.tag == "Q":
                element.Q = fitparams[element.idx]
                element.n = fitparams[element.idx2] #idx2 is generated on component creation and does not need to be passed to the component in its __init__ method
            if element.tag == "O":
                element.O = fitparams[element.idx]
                element.k = fitparams[element.idx2]
            if element.tag == "S":
                element.S = fitparams[element.idx]
                element.l = fitparams[element.idx2]
            if element.tag == "G":
                element.G = fitparams[element.idx]
                element.m = fitparams[element.idx2]
            if element.tag == "H":
                element.G = fitparams[element.idx]
                element.m = fitparams[element.idx2]
                element.beta = fitparams[element.idx3]
                element.gamma = fitparams[element.idx4]

    def impedance(self, fp, freq):
        """Impedance function for the circuit. This is also where the program check if the standard impedance function self.diagram.Z is being overridden by the custom impedance function ecm.custom_model.
        
        Arguments:
        self
        fp -- list of fit parameters
        freq -- NumPy array of frequencies
        
        Returns:
        NumPy array of complex impedances."""
        self.set_element_values(fp)
        if ecmcm.override_impedance_method:
            return ecmcm.custom_model_diagrams[ecmcm.custom_model_name][1](fp, freq)
        return self.diagram.Z(freq)

###############################
##HELPER CLASS FOR DECiM CORE##
###############################

class CircuitManager():
    def __init__(self, circuit = Circuit()):
        """Circuit manager for DECiM core.
        
        Init arguments:
        circuit -- Circuit object
        
        Attributes:
        circuit -- Circuit object
        
        Methods:
        changeCircuit -- change the circuit diagram"""
        self.circuit = circuit

    #Selection of a new equivalent circuit model
    def changeCircuit(self, cto):
        """Change the circuit diagram.
        
        Arguments:
        self
        cto -- new circuit diagram"""
        self.circuit.diagram = cto

###################
##CIRCUIT BUILDER##
###################

class PannableCanvas(tk.Canvas): #A canvas that can be panned and zoomed
    def __init__(self, master, selector_window, width = 600, height = 600):
        """Pannable canvas for drawing circuit diagrams.
        
        Init arguments:
        master -- master Frame, or None
        selector_window -- CircuitDefinitionWindow object
        width -- width of canvas
        height -- height of canvas
        
        Attributes:
        selector_window -- CircuitDefinitionWindow object
        element_width -- width of elements drawn on canvas
        element_height -- height of elements drawn on canvas
        element_horizontal_separation -- horizontal distance between elements
        element_vertical_separation -- vertical distance between elements
        horizontal_text_pad -- horizontal offset of text from the element position
        vertical_text_pad -- vertical offset of text from the element position
        line_width -- width of drawn lines
        text_anchor -- text anchor position
        font_info -- tuple of (font name, font size)
        fill_colour -- color string for element backgrounds
        line_colour -- color string for lines
        highlight_colour -- highlight color string
        alt_highlight_colour -- alternative highlight color string
        drawing_mode -- string, parallel or series
        merge_mode -- Boolean, indicates if Units are being merged or not
        draw_methods -- dict linking element tags to drawing methods for all the different elements
        elements_on_canvas -- dict linking elements (keys) to coordinates (values)
        dots -- list of places where new elements can be drawn; these are (x, y) tuples
        merge_points -- dict linking coordinates to pairs of units that can be merged
        
        Methods:
        Drawing of different elements: draw_box, draw_capacitor, draw_cpe, draw_gerischer, draw_inductor, draw_resistor, draw_wopen, draw_wshort
        
        Decision trees: is_first_element, is_nested_element, in_merge_mode, in_parallel_mode, in_series_mode, nearest_dot, nearest_merge_point, element_above_or_below_is_in_parallel_unit, element_left_or_right_is_in_series_unit
        
        Connections: draw_element_connection, find_maximum_extent, draw_parallel_connections, draw_series_connections, draw_unit_connection
        
        choose_dot_and_draw_dots -- draw all valid drawing positions and if in merge mode, highlight the closest merge point
        update_merge_points -- update self.merge_points
        generate_dots -- update self.dots
        draw_element -- determine if an element can be drawn in a given position, and in what unit that puts it; if in merge mode, decide which units are being merged
        redraw_circuit -- clear and redraw the circuit"""
        super().__init__(master = master, width = width, height = height)
        self.selector_window = selector_window
        
        self.bind("<ButtonPress-1>", self.draw_element) #Left click to add new element
        self.bind("<Motion>", self.choose_dot_and_draw_dots)
        #Bindings below taken from https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan
        self.bind("<ButtonPress-2>", lambda event: self.scan_mark(event.x, event.y)) #Right click to pan
        self.bind("<B2-Motion>", lambda event: self.scan_dragto(event.x, event.y, gain=1))
        self.bind("<ButtonPress-3>", lambda event: self.scan_mark(event.x, event.y)) #Wheel click to pan
        self.bind("<B3-Motion>", lambda event: self.scan_dragto(event.x, event.y, gain=1))
        
        #Drawing parameters
        self.element_width = 100
        self.element_height = 30
        self.element_horizontal_separation = 50
        self.element_vertical_separation = 100
        self.horizontal_text_pad = 40
        self.vertical_text_pad = 30
        self.line_width = 2
        self.text_anchor = tk.CENTER
        self.font_info = ("Verdana", 14)
        self.fill_colour = "#fff"
        self.line_colour = "#000"
        self.highlight_colour = "#00f"
        self.alt_highlight_colour = "#f00"
        
        #Drawing modes
        self.drawing_mode = "parallel"
        self.merge_mode = False
        
        #Drawing method dictionary: associating tags with drawing methods
        self.draw_methods = {"R": self.draw_resistor, "C": self.draw_capacitor, "L": self.draw_inductor, "Q": self.draw_cpe, "O": self.draw_wopen, "S": self.draw_wshort, "G": self.draw_gerischer, "H": self.draw_havriliak_negami}
        
        #Elements and their positions
        self.elements_on_canvas = {} #Keys are elements, values are coordinates
        self.dots = [] #Coordinates of places where new elements can be drawn
        
        #Merge positions for units
        self.merge_points = {} #Clicking a merge point merges two units. Here, keys are coordinates, and values are tuples of units.
    
    #Draw the network of dots that represent the positions in which drawing is possible.
    def choose_dot_and_draw_dots(self, event):
        """Draw all positions in which elements can be placed and the points on which the user can click to merge units.
        
        Arguments:
        self
        event -- mouse click event"""
        for d in self.dots:
            self.create_line(d[0] + self.element_width/2, d[1] - 5 + self.element_height/2, d[0] + self.element_width/2, d[1] + 5 + self.element_height/2, fill = "#000", width = self.line_width)
            self.create_line(d[0] - 5 + self.element_width/2, d[1] + self.element_height/2, d[0] + 5 + self.element_width/2, d[1] + self.element_height/2, fill = "#000", width = self.line_width)
        if self.merge_mode and len(self.merge_points) > 0:
            mindist = 9999999999999999
            mc = list(self.merge_points)[0]
            for m in self.merge_points:
                sqdist = (event.x - m[0])**2 + (event.y - m[1])**2
                if sqdist < mindist:
                    mindist = sqdist
                    mc = m
            for m in self.merge_points:
                if m != mc:
                    self.create_oval(m[0] - 5, m[1] - 5, m[0] + 5, m[1] + 5, fill = self.fill_colour, outline = self.line_colour, width = 2)
            self.create_oval(mc[0] - 5, mc[1] - 5, mc[0] + 5, mc[1] + 5, fill = self.alt_highlight_colour, outline = self.line_colour, width = 2)
            for u in self.selector_window.units:
                for e in u.list_elements():
                    if e.tag != "U":
                        self.draw_methods[e.tag](self.elements_on_canvas[e][0], self.elements_on_canvas[e][1], e.name)
            for u in self.merge_points[mc]:
                for e in u.list_elements():
                    if e.tag != "U":
                        self.draw_methods[e.tag](self.elements_on_canvas[e][0], self.elements_on_canvas[e][1], e.name, highlight = True)
            
    #Draw the points between units that, when clicked, will merge them.
    def update_merge_points(self):
        """Update the merge_points dictionary."""
        self.merge_points = {} #Clear the merge points
        xcen, ycen = [], [] #Centres of mass of the units
        for u in self.selector_window.units:
            xu, yu = [], []
            for e in u.list_elements():
                xu.append(self.elements_on_canvas[e][0])
                yu.append(self.elements_on_canvas[e][1])
            xcen.append(sum(xu)/len(xu))
            ycen.append(sum(yu)/len(yu))
        #All pairs of units are added into a dictionary, so they can be merged
        for u in range(len(self.selector_window.units)):
            for v in range(len(self.selector_window.units)):
                if u < v and len(self.selector_window.units[u].parent_units) == 0 and len(self.selector_window.units[v].parent_units) == 0: #Don't check units against themselves (u < v) and check that units are not subunits already
                    self.merge_points[((xcen[u] + xcen[v])/2, (ycen[u] + ycen[v])/2)] = (self.selector_window.units[u], self.selector_window.units[v]) #The point between the centres of mass is the point & the key
        
    #Box drawing, with some lines sticking out to act as connectors.
    def draw_box(self, x, y, linehighlight = False):
        """Draw a rectangle with two lines sticking out.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        linehighlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if linehighlight:
            lcol = self.alt_highlight_colour
        self.create_line(x, y + 0.5*self.element_height, x + 0.05*self.element_width, y + 0.5*self.element_height, fill = lcol, width = self.line_width)
        self.create_rectangle(x + 0.05*self.element_width, y, x + 0.95*self.element_width, y + self.element_height, outline = lcol, fill = self.fill_colour, width = self.line_width)
        self.create_line(x + 0.95*self.element_width, y + 0.5*self.element_height, x + self.element_width, y + 0.5*self.element_height, fill = lcol, width = self.line_width)
        
    #Drawing functions for the different circuit elements
    def draw_resistor(self, x, y, resistor_name, highlight = False):
        """Draw a resistor.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        resistor_name -- name of resistor
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.draw_box(x, y, linehighlight = highlight)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = resistor_name, fill = lcol)
        
    def draw_capacitor(self, x, y, capacitor_name, highlight = False):
        """Draw a capacitor.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        capacitor_name -- name of capacitor
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.create_line(x, y + 0.5*self.element_height, x + 0.45*self.element_width, y + 0.5*self.element_height, fill = lcol, width = self.line_width)
        self.create_line(x + 0.45*self.element_width, y, x + 0.45*self.element_width, y + self.element_height, fill = lcol, width = 2*self.line_width)
        self.create_line(x + 0.55*self.element_width, y, x + 0.55*self.element_width, y + self.element_height, fill = lcol, width = 2*self.line_width)
        self.create_line(x + 0.55*self.element_width, y + 0.5*self.element_height, x + self.element_width, y + 0.5*self.element_height, fill = lcol, width = self.line_width)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = capacitor_name, fill = lcol)
        
    def draw_inductor(self, x, y, inductor_name, highlight = False):
        """Draw an inductor.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        inductor_name -- name of inductor
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.create_line(x, y + 0.5*self.element_height, x + 0.05*self.element_width, y + 0.5*self.element_height, fill = lcol, width = self.line_width)
        arcno = 5
        for i in range(arcno):
            self.create_oval(x + (i + 0.15)*((self.element_width - 0.1)/arcno), y + self.element_height, x + (i + 0.85)*((self.element_width - 0.1)/arcno), y, outline = lcol, width = 2*self.line_width)
            if i < arcno - 1:
                self.create_oval(x + (i + 0.5 + 0.15)*((self.element_width - 0.1)/arcno), y + self.element_height, x + (i + 0.5 + 0.85)*((self.element_width - 0.1)/arcno), y, outline = lcol, width = 2*self.line_width)
        self.create_line(x + 0.95*self.element_width, y + 0.5*self.element_height, x + self.element_width, y + 0.5*self.element_height, fill = lcol, width = self.line_width)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = inductor_name, fill = lcol)
        
    def draw_cpe(self, x, y, cpe_name, highlight = False):
        """Draw a CPE.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        cpe_name -- name of inductor
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.draw_box(x, y, linehighlight = highlight)
        self.create_text(x + self.horizontal_text_pad, y + 0.5*self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = "CPE", fill = lcol)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = cpe_name, fill = lcol)
        
    def draw_wopen(self, x, y, wopen_name, highlight = False):
        """Draw a W_O element.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        wopen_name -- name of element
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.draw_box(x, y, linehighlight = highlight)
        self.create_text(x + self.horizontal_text_pad, y + 0.5*self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = "W (O)", fill = lcol)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = wopen_name, fill = lcol)
        
    def draw_wshort(self, x, y, wshort_name, highlight = False):
        """Draw a W_S element.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        wshort_name -- name of element
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.draw_box(x, y, linehighlight = highlight)
        self.create_text(x + self.horizontal_text_pad, y + 0.5*self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = "W (S)", fill = lcol)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = wshort_name, fill = lcol)
        
    def draw_gerischer(self, x, y, gerischer_name, highlight = False):
        """Draw a Gerischer element.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        gerischer_name -- name of element
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.draw_box(x, y, linehighlight = highlight)
        self.create_text(x + self.horizontal_text_pad, y + 0.5*self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = " G ", fill = lcol)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = gerischer_name, fill = lcol)
        
    def draw_havriliak_negami(self, x, y, havriliak_negami_name, highlight = False):
        """Draw a Havriliak-Negami element.
        
        Arguments:
        x -- horizontal coordinate
        y -- vertical coordinate
        gerischer_name -- name of element
        
        Keyword arguments:
        highlight -- Boolean, indicates if the lines should have the normal color or the highlight color"""
        lcol = self.line_colour
        if highlight:
            lcol = self.alt_highlight_colour
        self.draw_box(x, y, linehighlight = highlight)
        self.create_text(x + self.horizontal_text_pad, y + 0.5*self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = " H ", fill = lcol)
        self.create_text(x + self.horizontal_text_pad, y - self.vertical_text_pad, anchor = self.text_anchor, font = self.font_info, text = havriliak_negami_name, fill = lcol)
        
    #Drawing functions for connections
    #This depends on the type of connection (series or parallel) and which units the elements belong to. Two functions are needed: one for elements and one for units (merge mode)
    
    #For elements
    
    def draw_element_connection(self, elem1, elem2):
        """Draw lines to connect two elements.
        
        Arguments:
        elem1 -- circuit element
        elem2 -- circuit element"""
        pos1, pos2 = self.elements_on_canvas[elem1], self.elements_on_canvas[elem2]
        if elem1.parent_units[0] == elem2.parent_units[0] and elem1.parent_units[0].arrangement == "parallel":
            self.create_line(pos1[0], pos1[1] + self.element_height/2, pos2[0], pos2[1] + self.element_height/2, fill = self.line_colour, width = self.line_width)
            self.create_line(pos1[0] + self.element_width, pos1[1] + self.element_height/2, pos2[0] + self.element_width, pos2[1] + self.element_height/2, fill = self.line_colour, width = self.line_width)
        if elem1.parent_units[0] == elem2.parent_units[0] and elem1.parent_units[0].arrangement == "series":
            if pos1[0] - pos2[0] == self.element_width + self.element_horizontal_separation:
                self.create_line(pos1[0], pos1[1] + self.element_height/2, pos2[0] + self.element_width, pos2[1] + self.element_height/2, fill = self.line_colour, width = self.line_width)
            if pos2[0] - pos1[0] == self.element_width + self.element_horizontal_separation:
                self.create_line(pos2[0], pos2[1] + self.element_height/2, pos1[0] + self.element_width, pos1[1] + self.element_height/2, fill = self.line_colour, width = self.line_width)
    
    #For units
    
    def find_maximum_extent(self, u1, u2, direction = "x"):
        """For two units, find how for they extend in the x or y directions.
        
        Arguments:
        u1 -- Unit 1
        u2 -- Unit 2
        
        Keyword arguments:
        direction -- string, x or y
        
        Returns: tuple of:
        min1 -- Unit 1 minimum
        max1 -- Unit 1 maximum
        min2 -- Unit 2 minimum
        max2 -- Unit 2 maximum"""
        edir = {"x": 0, "y": 1}
        min1, max1, min2, max2 = 100000, -100000, 100000, -100000
        for e in u1.list_elements():
            if self.elements_on_canvas[e][edir[direction]] < min1:
                min1 = self.elements_on_canvas[e][edir[direction]]
            if self.elements_on_canvas[e][edir[direction]] > max1:
                max1 = self.elements_on_canvas[e][edir[direction]]
        for e in u2.list_elements():
            if self.elements_on_canvas[e][edir[direction]] < min2:
                min2 = self.elements_on_canvas[e][edir[direction]]
            if self.elements_on_canvas[e][edir[direction]] > max2:
                max2 = self.elements_on_canvas[e][edir[direction]]
        return min1, max1, min2, max2
    
    def draw_parallel_connections(self, u1, u2, direction = "x"):
        """Draw lines to connect two units in parallel.
        
        Arguments:
        u1 -- Unit 1
        u2 -- Unit 2
        
        Keyword arguments:
        direction -- string, x or y"""
        edir = {"x": 0, "y": 1}
        min1, max1, min2, max2 = self.find_maximum_extent(u1, u2, direction = direction)
        if direction == "x":
            y1, y2 = self.elements_on_canvas[u1.list_elements()[0]][1] + self.element_height/2, self.elements_on_canvas[u2.list_elements()[0]][1] + self.element_height/2
            if max1 > max2: #Extend both units equally far and connect the ends
                self.create_line(max2 + self.element_width, y2, max1 + self.element_width, y2, fill = self.line_colour, width = self.line_width)
                self.create_line(max1 + self.element_width, y1, max1 + self.element_width, y2, fill = self.line_colour, width = self.line_width)
            else:
                self.create_line(max1 + self.element_width, y1, max2 + self.element_width, y1, fill = self.line_colour, width = self.line_width)
                self.create_line(max2 + self.element_width, y1, max2 + self.element_width, y2, fill = self.line_colour, width = self.line_width)
            if min1 < min2:
                self.create_line(min1, y2, min2, y2, fill = self.line_colour, width = self.line_width)
                self.create_line(min1, y1, min1, y2, fill = self.line_colour, width = self.line_width)
            else:
                self.create_line(min2, y1, min1, y1, fill = self.line_colour, width = self.line_width)
                self.create_line(min2, y1, min2, y2, fill = self.line_colour, width = self.line_width)
        elif direction == "y":
            x1, x2 = self.elements_on_canvas[u1.list_elements()[0]][0] + self.element_width, self.elements_on_canvas[u2.list_elements()[0]][0]
            if max1 > max2: #Extend both units equally far and connect the ends
                self.create_line(x2, max1 + self.element_height/2, x2, max2 + self.element_height/2, fill = self.line_colour, width = self.line_width)
                self.create_line(x1, max1 + self.element_height/2, x2, max1 + self.element_height/2, fill = self.line_colour, width = self.line_width)
            else:
                self.create_line(x1, max1 + self.element_height/2, x1, max2 + self.element_height/2, fill = self.line_colour, width = self.line_width)
                self.create_line(x1, max2 + self.element_height/2, x2, max2 + self.element_height/2, fill = self.line_colour, width = self.line_width)
            if min1 < min2:
                self.create_line(x2, min1 + self.element_height/2, x2, min2 + self.element_height/2, fill = self.line_colour, width = self.line_width)
                self.create_line(x1, min1 + self.element_height/2, x2, min1 + self.element_height/2, fill = self.line_colour, width = self.line_width)
            else:
                self.create_line(x1, min1 + self.element_height/2, x1, min2 + self.element_height/2, fill = self.line_colour, width = self.line_width)
                self.create_line(x1, min2 + self.element_height/2, x2, min2 + self.element_height/2, fill = self.line_colour, width = self.line_width)
    
    def draw_series_connections(self, u1, u2):
        """Draw lines to connect two units in parallel.
        
        Arguments:
        u1 -- Unit 1
        u2 -- Unit 2
        
        Keyword arguments:
        direction -- string, x or y"""
        xmin1, xmax1, xmin2, xmax2 = self.find_maximum_extent(u1, u2, direction = "x") #Get both the horizontal and vertical extents
        ymin1, ymax1, ymin2, ymax2 = self.find_maximum_extent(u1, u2, direction = "y")
        ycen1, ycen2 = (ymax1 + ymin1)/2 + self.element_height/2, (ymax2 + ymin2)/2 + self.element_height/2 #From the extents, determine the vertical centres. These are the y-coordinates of the points between which the series connection will be drawn
        if xmax1 < xmin2:
            self.create_line(xmax1 + self.element_width, ycen1, (xmax1 + xmin2 + self.element_width)/2, ycen1, fill = self.line_colour, width = self.line_width)
            self.create_line((xmax1 + xmin2 + self.element_width)/2, ycen1, (xmax1 + xmin2 + self.element_width)/2, ycen2, fill = self.line_colour, width = self.line_width)
            self.create_line((xmax1 + xmin2 + self.element_width)/2, ycen2, xmin2, ycen2, fill = self.line_colour, width = self.line_width)
        elif xmax2 < xmin1:
            self.create_line(xmax2 + self.element_width, ycen2, (xmax2 + xmin1 + self.element_width)/2, ycen2, fill = self.line_colour, width = self.line_width)
            self.create_line((xmax2 + xmin1 + self.element_width)/2, ycen2, (xmax2 + xmin1 + self.element_width)/2, ycen1, fill = self.line_colour, width = self.line_width)
            self.create_line((xmax2 + xmin1 + self.element_width)/2, ycen1, xmin1, ycen1, fill = self.line_colour, width = self.line_width)
        elif xmax1 >= xmin2:
            if ymin1 > ymax2:
                eheight = self.element_height
            else:
                eheight = -self.element_height
            self.create_line(xmax1 + self.element_width, ycen1, xmax1 + self.element_width + self.element_horizontal_separation/2, ycen1, fill = self.line_colour, width = self.line_width)
            self.create_line(xmax1 + self.element_width + self.element_horizontal_separation/2, ycen1, xmax1 + self.element_width + self.element_horizontal_separation/2, ymin1 - eheight/2, fill = self.line_colour, width = self.line_width)
            self.create_line(xmax1 + self.element_width + self.element_horizontal_separation/2, ymin1 - eheight/2, xmin2 - self.element_horizontal_separation/2, ymin1 - eheight/2, fill = self.line_colour, width = self.line_width)
            self.create_line(xmin2 - self.element_horizontal_separation/2, ymin1 - eheight/2, xmin2 - self.element_horizontal_separation/2, ycen2, fill = self.line_colour, width = self.line_width)
            self.create_line(xmin2 - self.element_horizontal_separation/2, ycen2, xmin2, ycen2, fill = self.line_colour, width = self.line_width)
        elif xmax2 >= xmin1:#and ymin2 > ymax1:
            if ymin2 > ymax1:
                eheight = self.element_height
            else:
                eheight = -self.element_height
            self.create_line(xmax2 + self.element_width, ycen2, xmax2 + self.element_width + self.element_horizontal_separation/2, ycen2, fill = self.line_colour, width = self.line_width)
            self.create_line(xmax2 + self.element_width + self.element_horizontal_separation/2, ycen2, xmax2 + self.element_width + self.element_horizontal_separation/2, ymin2 - eheight/2, fill = self.line_colour, width = self.line_width)
            self.create_line(xmax2 + self.element_width + self.element_horizontal_separation/2, ymin1 - eheight/2, xmin1 - self.element_horizontal_separation/2, ymin2 - eheight/2, fill = self.line_colour, width = self.line_width)
            self.create_line(xmin1 - self.element_horizontal_separation/2, ymin2 - eheight/2, xmin1 - self.element_horizontal_separation/2, ycen1, fill = self.line_colour, width = self.line_width)
            self.create_line(xmin1 - self.element_horizontal_separation/2, ycen1, xmin1, ycen1, fill = self.line_colour, width = self.line_width)
                
    def draw_unit_connection(self, unit1, unit2):
        """Connect two units. Determine if they can be connected and if they can be, call the correct connection drawing function.
        
        Arguments:
        unit1 -- Unit 1
        unit2 -- Unit 2"""
        connection_arrangement = "parallel" #How to connect the units
        if len(unit1.parent_units) == 0 or len(unit2.parent_units) == 0: #Check that both units have at least one parent unit
            return
        if unit1.parent_units[0] != unit2.parent_units[0]: #Units to be connected must be in the same parent unit
            return
        else:
            connection_arrangement = unit1.parent_units[0].arrangement
        #Connecting two series units in parallel
        if unit1.arrangement == "series" and unit2.arrangement == "series" and connection_arrangement == "parallel":
            self.draw_parallel_connections(unit1, unit2,  direction = "x")
        #Connecting two parallel units in parallel
        elif unit1.arrangement == "parallel" and unit2.arrangement == "parallel" and connection_arrangement == "parallel":
            self.draw_parallel_connections(unit1, unit2, direction = "y")
        #Connecting a series unit and a parallel unit in parallel
        elif ((unit1.arrangement == "series" and unit2.arrangement == "parallel") or (unit1.arrangement == "parallel" and unit2.arrangement == "series")) and connection_arrangement == "parallel":
            if unit1.arrangement == "parallel" and unit2.arrangement == "series": #Treating the two cases as the same: flip the units
                u1, u2 = unit2, unit1
            else:
                u1, u2 = unit1, unit2
            self.draw_parallel_connections(u1, u2, direction = "x")
        #Connecting two series units in series. This is independent of the subunits' arrangements, unlike parallel connections
        if connection_arrangement == "series":
            if unit1.arrangement == "parallel" and unit2.arrangement == "series": #Treating the two cases as the same: flip the units
                u1, u2 = unit2, unit1
            else:
                u1, u2 = unit1, unit2
            self.draw_series_connections(u1, u2)
        
    #Logic functions to make the decision tree for drawing more readable.
    def is_first_element(self):
        """Check if the element being placed is the first element being put on the canvas.
        
        Returns:
        bool"""
        return len(self.selector_window.units) == 1 and len(self.selector_window.active_unit.elements) == 0
        
    def in_parallel_mode(self):
        """Return True if in parallel mode"""
        return self.drawing_mode == "parallel"
        
    def in_series_mode(self):
        """Return True if in series mode"""
        return self.drawing_mode == "series"
        
    def is_nested_element(self, element):
        """Check if an element exists in multiple units (this means that the element was originally placed in a unit which is now part of another unit). This is needed for the drawing rules.
        
        Arguments:
        self
        element -- a circuit element
        
        Returns:
        True if element is multiple units, False otherwise"""
        return len(element.parent_units) > 1
        
    def in_merge_mode(self):
        """Return True if in merge mode"""
        return self.merge_mode
        
    def nearest_dot(self, x, y):
        """Provide the dot on which an element can be drawn which is closest to a given coordinate. The given coordinate is usually a mouse position.
        
        Arguments:
        self
        x -- x coordinate
        y -- y coordinate
        
        Returns:
        dot -- tuple of (x, y)"""
        if len(self.dots) >= 1:
            dc = self.dots[0]
            mindist = (x - self.dots[0][0])**2 + (y - self.dots[0][1])**2
            for d in self.dots:
                sqdist = (x - d[0])**2 + (y - d[1])**2
                if sqdist < mindist:
                    mindist = sqdist
                    dc = d
            return dc
        return None
        
    def nearest_merge_point(self, x, y):
        """Provide the merge point which is closest to a given coordinate. The given coordinate is usually a mouse position.
        
        Arguments:
        self
        x -- x coordinate
        y -- y coordinate
        
        Returns:
        merge point -- tuple of (x, y)"""
        mdots = list(self.merge_points)
        if len(mdots) > 0:
            dc = mdots[0]
            mindist = (x - dc[0])**2 + (y - dc[1])**2
            for d in mdots:
                sqdist = (x - d[0])**2 + (y - d[1])**2
                if sqdist < mindist:
                    mindist = sqdist
                    dc = d
        return dc
        
    def element_above_or_below_is_in_parallel_unit(self, dot):
        """Check if an element exists above or below a given valid drawing site (dot) and if it is in parallel mode.
        
        Arguments:
        self
        dot -- (x, y) tuple
        
        Returns: tuple of:
        Boolean
        Element in above/below position that is in parallel mode (or None if the Boolean is False)"""
        for e in self.elements_on_canvas:
            if self.elements_on_canvas[e][1] == dot[1] - self.element_vertical_separation - self.element_height and self.elements_on_canvas[e][0] == dot[0] and e.parent_units[-1].arrangement == "parallel" and not self.is_nested_element(e):
                return True, e
            elif self.elements_on_canvas[e][1] == dot[1] + self.element_vertical_separation + self.element_height and self.elements_on_canvas[e][0] == dot[0] and e.parent_units[-1].arrangement == "parallel" and not self.is_nested_element(e):
                return True, e
        return False, None
                
    def element_left_or_right_is_in_series_unit(self, dot):
        """Check if an element exists to the left or right of a given valid drawing site (dot) and if it is in series mode.
        
        Arguments:
        self
        dot -- (x, y) tuple
        
        Returns: tuple of:
        Boolean
        Element in left/right position that is in series mode (or None if the Boolean is False)"""
        for e in self.elements_on_canvas:
            if self.elements_on_canvas[e][0] == dot[0] + self.element_horizontal_separation + self.element_width and self.elements_on_canvas[e][1] == dot[1] and e.parent_units[-1].arrangement == "series" and not self.is_nested_element(e):
                return True, e
            elif self.elements_on_canvas[e][0] == dot[0] - self.element_horizontal_separation - self.element_width and self.elements_on_canvas[e][1] == dot[1] and e.parent_units[-1].arrangement == "series" and not self.is_nested_element(e):
                return True, e
        return False, None
    
    def generate_dots(self):
        """(Re)generate the self.dots list of valid drawing positions based on the positions of all currently existing circuit elements. In principle, any unoccupied position that is element_horizontal_separation + element_width to the side of an element or element_vertical_separation + element_height above or below an element is valid."""
        self.dots = []
        potential_dots = []
        for i in self.elements_on_canvas:
            potential_dots.append(self.elements_on_canvas[i])
        for i in self.elements_on_canvas:
            for k in [(self.elements_on_canvas[i][0] - self.element_horizontal_separation - self.element_width, self.elements_on_canvas[i][1]), (self.elements_on_canvas[i][0] + self.element_horizontal_separation + self.element_width, self.elements_on_canvas[i][1]), (self.elements_on_canvas[i][0], self.elements_on_canvas[i][1] - self.element_vertical_separation - self.element_height), (self.elements_on_canvas[i][0], self.elements_on_canvas[i][1] + self.element_vertical_separation + self.element_height)]:
                if k not in potential_dots:
                    self.dots.append(k)
    
    def draw_element(self, event):
        """Handle all the checks when a user clicks on the canvas to place a new element or to merge units. This function is responsible for adding the new element or Unit to the circuit, drawing the new element and/or new connections and updating the canvas.
        
        Arguments:
        self
        event -- mouse click event, which contains the (x, y) coordinates on which the user clicked."""
        self.selector_window.make_backups() #Save the previous state
        self.selector_window.error_message.set("") #Clear any errors related to previous draw move
        cursor_x, cursor_y = self.canvasx(event.x), self.canvasy(event.y) #Mouse coordinates
        element_tag = self.selector_window.chosen_element.get()[0] #Element tag/abbreviation: R, C, L, ...
        element_name = element_tag + str(self.selector_window.element_numbers[element_tag]) #Full name of the element: R0, C0, ...
        #The first element is drawn at the cursor position.
        if self.is_first_element():
            if self.in_series_mode(): #For the first element, it is important that the arrangement of the first unit is CHANGED. For subsequent elements, this does not matter.
                self.selector_window.active_unit.arrangement = "series"
            if self.in_parallel_mode():
                self.selector_window.active_unit.arrangement = "parallel"
            self.selector_window.add_element() #Add the element to the circuit
            self.elements_on_canvas[self.selector_window.active_unit.elements[-1]] = (cursor_x, cursor_y) #Save the coordinates of the element, with the element itself as a key in a dictionary and the coordinates (tuple) as the value.
            self.redraw_circuit() #Draw the first element
            return #Stop immediately
        #If there already are elements, then create a spot available for drawing next to each of them, barring already occupied coordinates.
        elif not self.is_first_element() and not self.in_merge_mode():
            self.generate_dots()
            #The mouse will snap to the closest available coordinate, which will light up. This is detailed in the self.choose_dot method.
            #The nearest dot to the mouse coordinates is where the new element is drawn.
            draw_position = self.nearest_dot(cursor_x, cursor_y)
            #Check which elements are nearby and if they can be connected to
            parallel_check = self.element_above_or_below_is_in_parallel_unit(draw_position) #Check if there is an element in a parallel unit above or below the draw position. Save the check result and, if True, the found element.
            series_check = self.element_left_or_right_is_in_series_unit(draw_position) #Check if there is an element in a series unit to the left or right of the draw position. Save the check result and, if True, the found element.
        #If drawing in parallel, then the positions above and below the elements in parallel units extend their units with new elements. The left and right positions will instead create a new parallel unit in which the new element will be placed.
        if not self.is_first_element() and not self.in_merge_mode() and self.in_parallel_mode() and parallel_check[0]:
            self.selector_window.active_unit = parallel_check[1].parent_units[-1] #Set the nearby parallel element's unit active
            self.selector_window.add_element() #Add the new element to the active unit
            self.elements_on_canvas[self.selector_window.active_unit.elements[-1]] = draw_position #The element is added to the coordinate dictionary
        elif not self.is_first_element() and not self.in_merge_mode() and self.in_parallel_mode() and not parallel_check[0]:
            self.selector_window.new_parallel_unit() #Make a new series unit. This unit is automatically active
            self.selector_window.add_element() #Add the new element to the active unit
            self.elements_on_canvas[self.selector_window.active_unit.elements[-1]] = draw_position #The element is added to the coordinate dictionary
        #If drawing in series, then the positions to the left and right of the elements in series units extend their units with new elements. The above and below positions will instead create a new series unit in which the new element will be placed.
        elif not self.is_first_element() and not self.in_merge_mode() and self.in_series_mode() and series_check[0]:
            self.selector_window.active_unit = series_check[1].parent_units[-1] #Set the closest element's unit active
            self.selector_window.add_element() #Add the new element to the active unit
            self.elements_on_canvas[self.selector_window.active_unit.elements[-1]] = draw_position #The element is added to the coordinate dictionary
        elif not self.is_first_element() and not self.in_merge_mode() and self.in_series_mode() and not series_check[0]:
            self.selector_window.new_series_unit() #Make a new series unit. This unit is automatically active
            self.selector_window.add_element() #Add the new element to the active unit
            self.elements_on_canvas[self.selector_window.active_unit.elements[-1]] = draw_position #The element is added to the coordinate dictionary
        #If in merge mode, then units will be merged based on the drawing mode (series/parallel) and the mouse position.
        elif self.in_merge_mode():
            to_be_merged = self.merge_points[self.nearest_merge_point(cursor_x, cursor_y)] #The tuple of units to be merged
            #Check if the units being merged are not 1) being merged in series while they contain elements that are only vertically separated or 2) being merged in parallel while they contain elements that are only horizontally separated
            e1_x, e1_y, e2_x, e2_y = [], [], [], []
            merge_allowed = True
            for e1 in to_be_merged[0].list_elements():
                e1_x.append(self.elements_on_canvas[e1][0])
                e1_y.append(self.elements_on_canvas[e1][1])
            for e2 in to_be_merged[1].list_elements():
                e2_x.append(self.elements_on_canvas[e2][0])
                e2_y.append(self.elements_on_canvas[e2][1])
            for i in range(len(e1_x)):
                for k in range(len(e2_x)):
                    if self.in_parallel_mode() and e1_y[i] == e2_y[k]:
                        merge_allowed = False
                    if self.in_series_mode() and e1_x[i] == e2_x[k]:
                        merge_allowed = False
            if merge_allowed:
                self.selector_window.merge_units(self.merge_points[self.nearest_merge_point(cursor_x, cursor_y)], self.drawing_mode) #Merge units. The drawing mode determines the arrangement, as with elements. The elements associated with the dot in self.mdots (the closest elements) are merged.
            else:
                self.selector_window.error_message.set("Merge failed: units\nnot separated correctly")
            self.merge_mode = False #Turn off merge mode
        self.redraw_circuit() #Draw all elements and connections
        self.selector_window.update_mode_label()
        self.generate_dots()
        self.choose_dot_and_draw_dots(event)
        
    def redraw_circuit(self):
        """Clear the canvas and redraw the circuit."""
        self.delete("all") #Clear the canvas
        elems = list(self.elements_on_canvas)
        for e1 in range(len(elems)): #Draw all the connections between elements within units
            for e2 in range(e1, len(elems), 1):
                self.draw_element_connection(elems[e1], elems[e2])
        for u in range(len(self.selector_window.units)): #Draw connections between units
            for v in range(len(self.selector_window.units)):
                if u < v:
                    self.draw_unit_connection(self.selector_window.units[u], self.selector_window.units[v])
        for element in self.elements_on_canvas: #Draw all the elements
            element_position = self.elements_on_canvas[element]
            self.draw_methods[element.tag](element_position[0], element_position[1], element.name)

class CircuitDefinitionWindow(tk.Toplevel): #The window that is lauched for making changes to the equivalent circuit model.
    def __init__(self, previous_circuit):
        """Circuit selection window class. Contains all the circuit building code that is not tied to the canvas.
        
        Init arguments:
        self
        previous_circuit -- the previous Circuit object being used by DECiM core
        
        Attributes:
        width -- window width
        height -- window height
        
        chosen_circuit -- the Circuit object that DECiM core will accept as the new circuit
        active_unit -- the Unit to which elements are currently being added
        units -- list of all Units
        old_circuit -- the previous Circuit object being used by DECiM core
        element_idx -- the current element index
        element_numbers -- the current numbers of all individual element types
        
        previously_drawn_circuits -- list of Circuit objects created in previous steps (merges, element placements)
        previous_active_unit_numbers -- list of numbers identifying Unit objects that were previously the active unit
        previous_element_numbers -- list of previous dicts of element numbers
        previous_canvas_elements -- list of previous self.canvas.elements_on_canvas dicts
        previous_dots -- list of previous self.canvas.dots lists
        previous_merge_points -- list of previous self.canvas.merge_points dicts
        
        buttonframe -- tk.Frame onto which buttons are place
        canvasframe -- tk.Frame onto which the PannableCanvas is placed
        canvas -- PannableCanvas object
        
        new_elements -- list of elements for the user to choose from
        
        active_mode -- tk.StringVar indicating to the user what the drawing mode is
        merge_active -- tk.StringVar indicating to the user if merge mode is on
        mode_label -- tk.Label that displays active_mode
        merge_label -- tk.Label that displays merge_active
        
        element_dropdown_label -- label to indicate what the dropdown is for
        chosen_element -- StringVar to indicate which element has been chosen for drawing
        element_dropdown -- tk.OptionMenu containing the different circuit elements that can be chosen for drawing
        
        drawing_mode_button -- tk.Button for switching between series and parallel drawing modes
        merge_units_button -- button to toggle merge mode on or off
        undo_button -- button to undo most recent merge or placement; works multiple times
        clear_button -- button to clear the circuit and PannableCanvas
        end_button -- button to accept the circuit and close the circuit builder
        
        error_message -- StringVar to display an error message when an invalid draw move is made
        error_label -- label that displays error_message
        
        Methods:
        update_mode_label -- update mode_label and merge_active
        make_backups -- update the lists in which previous circuits, active units, etc. are held
        reset_element_dropdown -- add all the options in new_elements to element_dropdown
        toggle_series_parallel_mode -- switch drawing modes
        toggle_merge -- toggle merge mode on/off
        new_series_unit -- add a new series unit to units and make this unit the active_unit
        new_parallel_unit -- add a new parallel unit to units and make this unit the active_unit
        add_element -- add an element to the active_unit
        merge_units -- create a new unit and put the selected units into it
        set_circuit_to_biggest_unit -- set chosen_circuit.diagram to the Unit that contains the most elements
        undo_merge_or_placement -- undo the latest element placement or Unit merge
        clear_circuit -- clear chosen_circuit.diagram and clear the canvas
        terminate -- close the window; if chosen_circuit.diagram contains no elements, set it to the old circuit diagram"""
        #Window geometry
        super().__init__()
        self.title("Equivalent circuit model builder")
        self.width = int(0.75*self.winfo_screenwidth())
        self.height = int(0.75*self.winfo_screenheight())
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        
        #Clear the circuit, but save the old one; also make a list of previously drawn circuits for the Undo button
        self.chosen_circuit = Circuit(diagram = Unit(0, arrangement = "parallel"))
        self.chosen_circuit.diagram.elements = []
        self.active_unit = self.chosen_circuit.diagram
        self.units = [self.active_unit]
        self.old_circuit = previous_circuit
        self.previously_drawn_circuits = []
        self.previous_active_unit_numbers = []
        self.previous_element_numbers = []
        self.previous_canvas_elements = []
        self.previous_dots = []
        self.previous_merge_points = []
        
        #Tracking the circuit elements and units
        self.element_idx = 0
        self.element_numbers = {"R": 0, "C": 0, "L": 0, "Q": 0, "S": 0, "O": 0, "G": 0, "H": 0, "U": 0}
            
        #Create a frame for the buttons and for the canvas
        self.buttonframe = tk.Frame(self)
        self.canvasframe = tk.Frame(self, highlightbackground = "#000", highlightthickness = 2)
        self.buttonframe.pack(side = tk.LEFT, anchor = tk.N)
        self.canvasframe.pack(side = tk.LEFT, anchor = tk.N, expand = True, fill = tk.BOTH)
        
        #Putting the canvas on the canvas frame
        self.canvas = PannableCanvas(self.canvasframe, self, width = self.winfo_screenwidth(), height = self.winfo_screenheight())
        self.canvas.pack(side = tk.TOP, anchor = tk.W)
        
        #New elements to choose from
        self.new_elements = ["R: Resistor", "C: Capacitor", "L: Inductor", "Q: Constant Phase Element", "O: Warburg Open", "S: Warburg Short", "G: Gerischer Element", "H: Havriliak-Negami Element"]
        
        #Indicate the current drawing mode
        self.active_mode = tk.StringVar()
        self.merge_active = tk.StringVar()
        self.mode_label = tk.Label(self.buttonframe, textvariable = self.active_mode)
        self.mode_label.pack(side = tk.TOP, anchor = tk.W)
        self.merge_label = tk.Label(self.buttonframe, textvariable = self.merge_active)
        self.merge_label.pack(side = tk.TOP, anchor = tk.W)
        self.update_mode_label()
        
        #Element selection dropdown. Elements are selected here and can then be drawn on self.canvas
        self.element_dropdown_label = tk.Label(self.buttonframe, text = "Next element:")
        self.element_dropdown_label.pack(side = tk.TOP, anchor = tk.W)
        self.chosen_element = tk.StringVar()
        self.chosen_element.set("R: Resistor")
        self.element_dropdown = tk.OptionMenu(self.buttonframe, self.chosen_element, "R: Resistor")
        self.element_dropdown.pack(side = tk.TOP, anchor = tk.W)
        self.reset_element_dropdown()
        
        #Put the buttons on the button frame
        self.drawing_mode_button = tk.Button(self.buttonframe, text = "Series/parallel mode", command = self.toggle_series_parallel_mode)
        self.drawing_mode_button.pack(side = tk.TOP, anchor = tk.W)
        self.merge_units_button = tk.Button(self.buttonframe, text = "Merge units", command = self.toggle_merge)
        self.merge_units_button.pack(side = tk.TOP, anchor = tk.W)
        self.undo_button = tk.Button(self.buttonframe, text = "Undo previous change", command = self.undo_merge_or_placement)
        self.undo_button.pack(side = tk.TOP, anchor = tk.W)
        self.clear_button = tk.Button(self.buttonframe, text = "Reset circuit builder", command = self.clear_circuit)
        self.clear_button.pack(side = tk.TOP, anchor = tk.W)
        self.end_button = tk.Button(self.buttonframe, text = "Use circuit and close", command = self.terminate)
        self.end_button.pack(side = tk.TOP, anchor = tk.W)
        
        #Indicate drawing errors
        self.error_message = tk.StringVar()
        self.error_label = tk.Label(self.buttonframe, textvariable = self.error_message, fg = "#c00")
        self.error_label.pack(side = tk.TOP, anchor = tk.W)
        
    def update_mode_label(self):
        """Set the drawing mode and merge mode StringVars (active_mode and merge_active) to the appropriate values given the current drawing mode and whether or not merge mode is active."""
        if self.canvas.drawing_mode == "series":
            self.active_mode.set("Series mode")
        if self.canvas.drawing_mode == "parallel":
            self.active_mode.set("Parallel mode")
        if self.canvas.merge_mode:
            self.merge_active.set("Merge mode ON")
            self.merge_label.configure(fg = "#000")
        else:
            self.merge_active.set("Merge mode OFF")
            self.merge_label.configure(fg = "#aaa")
        
    def reset_element_dropdown(self):
        """Add all the element choices in new_elements to element_dropdown."""
        self.element_dropdown["menu"].delete(0, tk.END)
        for elem in self.new_elements:
            self.element_dropdown["menu"].add_command(label = elem, command = tk._setit(self.chosen_element, elem))
    
    def make_backups(self):
        """Save the current circuit, active unit, and element numbers so the Undo button can work."""
        self.previously_drawn_circuits.append(copy.deepcopy(self.chosen_circuit))
        self.previous_active_unit_numbers.append(copy.deepcopy(self.active_unit.number))
        self.previous_element_numbers.append(copy.deepcopy(self.element_numbers))
        self.previous_canvas_elements.append(copy.deepcopy(self.canvas.elements_on_canvas))
        self.previous_dots.append(copy.deepcopy(self.canvas.dots))
        self.previous_merge_points.append(copy.deepcopy(self.canvas.merge_points))
    
    def toggle_series_parallel_mode(self):
        """Change between series and parallel drawing modes."""
        if self.canvas.drawing_mode == "series":
            self.canvas.drawing_mode = "parallel"
        elif self.canvas.drawing_mode == "parallel":
            self.canvas.drawing_mode = "series"
        self.update_mode_label()
            
    def toggle_merge(self):
        """Toggle merge mode on or off."""
        if len(self.active_unit.list_elements()) > 0:
            self.canvas.merge_mode = not self.canvas.merge_mode
            self.update_mode_label()
            self.canvas.update_merge_points()
                
    def new_series_unit(self):
        """Create a new series unit, add it to units, make it the active_unit and increase the unit number."""
        self.units.append(Unit(self.element_numbers["U"], arrangement = "series"))
        self.active_unit = self.units[-1]
        self.element_numbers["U"] += 1
        
    def new_parallel_unit(self):
        """Create a new parallel unit, add it to units, make it the active_unit and increase the unit number."""
        self.units.append(Unit(self.element_numbers["U"], arrangement = "parallel"))
        self.active_unit = self.units[-1]
        self.element_numbers["U"] += 1
        
    def add_element(self):
        """Add an element to active_unit, set its value, index, and number, and increase the element index and number."""
        if self.chosen_element.get()[0] == "R":
            new_element = Resistor(1, self.element_numbers["R"], self.element_idx, self.active_unit)
            self.element_idx += 1
        if self.chosen_element.get()[0] == "C":
            new_element = Capacitor(1e-12, self.element_numbers["C"], self.element_idx, self.active_unit)
            self.element_idx += 1
        if self.chosen_element.get()[0] == "L":
            new_element = Inductor(1e-12, self.element_numbers["L"], self.element_idx, self.active_unit)
            self.element_idx += 1
        if self.chosen_element.get()[0] == "Q":
            new_element = ConstantPhaseElement(1e-12, 1, self.element_numbers["Q"], self.element_idx, self.active_unit)
            self.element_idx += 2
        if self.chosen_element.get()[0] == "S":
            new_element = WarburgShort(1, 1e-12, self.element_numbers["S"], self.element_idx, self.active_unit)
            self.element_idx += 2
        if self.chosen_element.get()[0] == "O":
            new_element = WarburgOpen(1, 1e-12, self.element_numbers["O"], self.element_idx, self.active_unit)
            self.element_idx += 2
        if self.chosen_element.get()[0] == "G":
            new_element = GerischerElement(1, 1e-12, self.element_numbers["G"], self.element_idx, self.active_unit)
            self.element_idx += 2
        if self.chosen_element.get()[0] == "H":
            new_element = HavriliakNegami(1, 1e-12, 1, 1, self.element_numbers["H"], self.element_idx, self.active_unit)
            self.element_idx += 4
        self.element_numbers[self.chosen_element.get()[0]] += 1
        self.active_unit.add_element(new_element)
            
    def merge_units(self, units, arrangement):
        """Merge multiple units into a new unit.
        
        Arguments:
        self
        units -- list of units to be merged
        arrangement -- series or parallel; the arrangement of the new unit into which the other units are merged"""
        if arrangement == "parallel": #Create a new unit with the appropriate arrangement
            self.new_parallel_unit()
        if arrangement == "series":
            self.new_series_unit()
        for u in units: #Add all units being merged into the new unit
            self.active_unit.add_element(u)
            u.parent_units.append(self.active_unit)
        for e in self.active_unit.list_elements(): #For every involved element, indicate that there is now a new parent unit by appending it to the list of parent units
            e.parent_units.append(self.active_unit)
    
    def set_circuit_to_biggest_unit(self):
        """Set chosen_circuit to a Circuit whose diagram is the largest unit in units. Used in the terminate method."""
        ulens = []
        for u in self.units:
            ulens.append(len(u.list_elements()))
        uidx = ulens.index(max(ulens))
        self.chosen_circuit = Circuit(diagram = self.units[uidx])
    
    def undo_merge_or_placement(self):
        """Undo the latest merge or element placement."""
        if len(self.previously_drawn_circuits) > 1:
            self.chosen_circuit = copy.deepcopy(self.previously_drawn_circuits[-1])
            del self.previously_drawn_circuits[-1]
            self.active_unit = self.chosen_circuit.diagram.find_unit(self.previous_active_unit_numbers[-1])
            del self.previous_active_unit_numbers[-1]
            self.element_numbers = copy.deepcopy(self.previous_element_numbers[-1])
            del self.previous_element_numbers[-1]
            self.canvas.elements_on_canvas = copy.deepcopy(self.previous_canvas_elements[-1])
            del self.previous_canvas_elements[-1]
            self.canvas.dots = copy.deepcopy(self.previous_dots[-1])
            del self.previous_dots[-1]
            self.canvas.merge_points = copy.deepcopy(self.previous_merge_points[-1])
            del self.previous_merge_points[-1]
            self.canvas.redraw_circuit()
        elif len(self.previously_drawn_circuits) == 1:
            self.clear_circuit()
    
    def clear_circuit(self):
        """Set chosen_circuit to a Circuit whose diagram is empty, reset units and active_unit, clear the canvas, clear the canvas's list of dots, clear the canvas's dictionary of elements, and clear the backup lists."""
        self.chosen_circuit = Circuit(diagram = Unit(0, arrangement = "parallel"))
        self.chosen_circuit.diagram.elements = []
        for n in self.element_numbers:
            self.element_numbers[n] = 0
        self.units = [self.chosen_circuit.diagram]
        self.active_unit = self.chosen_circuit.diagram
        self.canvas.delete("all")
        self.canvas.dots = []
        self.canvas.elements_on_canvas = {}
        self.canvas.merge_points = {}
        self.previously_drawn_circuits = []
        self.previous_active_unit_numbers = []
        self.previous_element_numbers = []
            
    def terminate(self):
        """Close the window and, if chosen_circuit is None or its diagram is empty, set chosen_circuit equal to old_circuit."""
        self.set_circuit_to_biggest_unit()
        if self.chosen_circuit.diagram == None:
            self.chosen_circuit = self.old_circuit
        elif len(self.chosen_circuit.diagram.list_elements()) == 0:
            print("No elements in new circuit. Reusing old circuit.")
            self.chosen_circuit = self.old_circuit
        self.destroy()