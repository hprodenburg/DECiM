"""Part of DECiM. This file contains all functions related to file I/O. Last modified 10 January 2023 by Henrik Rodenburg.

Functions:
parseData -- parse a text file containing measured data, return a dataSet
parseCircuitString -- parse a circuit string, return a Circuit
assignIndicesAndValues -- correct the indices and values obtained from a result (.recm2) file
parseResult -- parse an RECM2 file, return a Circuit, measured dataSet and list of model parameters
createResultFile -- create an RECM2 file based on the current data and model
parseCircuitPresets -- validate the circuit definitions in the circuit presets (ecm_presets.decim_circuits) file, return a list with correct circuit strings"""

###########
##IMPORTS##
###########

import numpy as np
import copy
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.simpledialog as sdiag
import tkinter.filedialog as fdiag

from ecm_datastructure import dataSet
from ecm_circuits import Circuit, Unit, Resistor, Capacitor, Inductor, ConstantPhaseElement, WarburgOpen, WarburgShort, GerischerElement

#####################
##DATA FILE PARSING##
#####################

def parseData(filename, reverse = True):
    """Read a data file. The assumed structure is three columns: linear frequency (Hz), real part of impedance (Ohm), imaginary part of impedance (Ohm).
    
    Non-keyword arguments:
    filename -- relative path to file and name of file
    
    Keyword arguments:
    reverse -- Boolean, reverse the order of the imported data if True. Default is True. Should be True if the input file begins with the highest frequencies, otherwise it should be False.
    
    Returns:
    ndata -- list of arrays. ndata[0] is the frequency, ndata[1] the real impedance and ndata[2] the imaginary impedance."""
    data = []
    file = open(filename, "r")
    for line in file:
        xq = False
        if filename[-4:] == ".csv" or filename[-5:-1] == ".csv":
            datum = line.split(",")
        else:
            datum = line.split()
        while len(datum) > len(data):
            data.append([])
        for d in range(len(datum)):
            try:
                data[d].append(float(datum[d]))
            except:
                if datum[d] in ["1.#QNAN", "nan", "NaN", "inf", "Inf"] and not xq: #Overload detection
                    data[0].pop(-1)
                    xq = True
                continue
    file.close()
    ndata = []
    if reverse:
        for d in data:
            d.reverse()
            ndata.append(np.array(d))
        return ndata
    return data

##################################
##PROCESSING RESULT FILE PARSING##
##################################

def parseCircuitString(circuit_string):
    """Parse a circuit string [such as (R0Q0)].
    
    Arguments:
    circuit_string -- the string defining a circuit
    
    Returns:
    ecm_circuits.Circuit object whose diagram matches the one described by the circuit string."""
    if circuit_string[0] != "{" and circuit_string[-1] != "}" and circuit_string[0] != "(" and circuit_string[-1] != ")": #Fix circuits with no surrounding matching brackets by assuming they are series units
        circuit_string = "{" + circuit_string + "}"
    mode_signs = {"(": "parallel", "{": "series"} #Linking of brackets to modes. Needed to correctly initialize new units when brackets are read.
    counts = {"R": 0, "C": 0, "L": 0, "Q": 0, "O": 0, "S": 0, "G": 0, "U": 0} #Keeping track of how many of each element there are.
    elements = {"R": Resistor, "C": Capacitor, "L": Inductor, "Q": ConstantPhaseElement, "O": WarburgOpen, "S": WarburgShort, "G": GerischerElement} #Classes associated with the letters in the string.
    level = 0 #The 'depth' of the current unit in the circuit. Basically how many open brackets it finds itself behind.
    eidx = 0 #The index in the parameter list associated with the element. This is incorrectly handled in this function, because further specification beyond the circuit string is needed to do it right. This is what the assignIndicesAndValues function handles.
    units = [] #Units. Units the elements are in.
    unit_merges = {} #Dictionary with unit numbers as keys and [unit, merge value [bool]] lists as values
    open_unit_numbers = []
    for pos in range(len(circuit_string)): #Loop over the indices to get direct access to the index of the character.
        c = circuit_string[pos] #Save the character.
        if c in ["(", "{"]: #Handling open brackets
            level += 1 #Level increases
            units.append(Unit(copy.deepcopy(counts["U"]), arrangement = mode_signs[c])) #New unit is made & added to units list
            unit_merges[units[-1].number] = [units[-1], True] #New unit is also added to merge dictionary
            open_unit_numbers = [] #Merging. First, collect the numbers of all still-open units
            for u in unit_merges: #Loop over all unit numbers in the merge dictionary
                if unit_merges[u][1] and unit_merges[u][0] != units[-1]: #If there is an open unit and it is NOT the new unit...
                    open_unit_numbers.append(u) #...save its number.
            if len(open_unit_numbers) > 0: #If there are open units...
                unit_merges[max(open_unit_numbers)][0].add_element(units[-1]) #...add the new unit to the highest-number open unit.
            counts["U"] += 1 #The unit count is updated
        elif c in ["R", "L", "C"]: #R, L, C are all handled the same way, since they have the same general structure in ecm_circuits.
            enum = counts[c] #By default, the element's number is equal to the current count of its type of element.
            if len(circuit_string) > pos: #However, if the circuit string contains element numbers, the element number in the string is used instead.
                if circuit_string[pos + 1] in "012345789":
                    enum = int(circuit_string[pos + 1])
            for k in reversed(list(unit_merges.keys())):
                if unit_merges[k][1]:
                    unit_merges[k][0].add_element(elements[c](1, enum, eidx, units[-1])) #Add element to most recently added, still open unit. The actual value of the resistor, capacitor or inductor is set to 1. It will be changed later, by the assignIndicesAndValues function.
                    break
            counts[c] += 1 #The count for the type of element is updated.
            eidx += 1 #The index is also updated (although it is wrong -- again, this is fixed by the assignIndicesAndValues function)
        elif c in ["Q", "O", "S", "G"]: #Q, O, S, G are all handled the same way, since they have the same general structure in ecm_circuits.
            enum = counts[c] #By default, the element's number is equal to the current count of its type of element.
            if len(circuit_string) > pos: #However, if the circuit string contains element numbers, the element number in the string is used instead.
                if circuit_string[pos + 1] in "012345789":
                    enum = int(circuit_string[pos + 1])
            for k in reversed(list(unit_merges.keys())):
                if unit_merges[k][1]:
                    unit_merges[k][0].add_element(elements[c](1, 1, enum, eidx, units[-1])) #Add element to most recently added, still open unit. The actual value of the resistor, capacitor or inductor is set to 1. It will be changed later, by the assignIndicesAndValues function.
                    break
            counts[c] += 1 #The count for the type of element is updated.
            eidx += 2 #The index is also updated (although it is wrong -- again, this is fixed by the assignIndicesAndValues function)
        elif c in [")", "}"]:
            open_unit_numbers = [] #Merging, part 2: closing units. First, collect the numbers of all still-open units
            for u in unit_merges: #Loop over all unit numbers in the merge dictionary
                if unit_merges[u][1]:
                    open_unit_numbers.append(u)
            if len(open_unit_numbers) > 0:
                unit_merges[max(open_unit_numbers)][1] = False
        elif c in ["\n", "\r", " ", "\t"]: #Stop searching when whitespace or linebreak characters are found
            break
    maxlen = 0
    full_circuit_unit = units[0]
    for u in units:
        if len(u.list_elements()) > maxlen:
            maxlen = len(u.list_elements())
            full_circuit_unit = u
    return Circuit(diagram = full_circuit_unit) #With all the units sorted out, it is time to return a circuit consisting of the lowest-level unit; this unit holds the whole circuit.

def assignIndicesAndValues(specification_lines, raw_units): #Take the specification lines (in the RECM2 file) and the newly generated units with wrong values and indices as input.
    """Change the indices and values of circuit elements based on input from an RECM2 file and return fit parameters that can be accessed by all parts of DECiM.
    
    Arguments:
    specification_lines -- part of the RECM2 file that contains information about the individual circuit elements
    raw_units -- ecm_circuits.Circuit object whose elements' values and indices must be corrected
    
    Returns:
    fit_parameters -- list of fit parameters"""
    fit_parameters = list(np.zeros(100) + 1) #List of model parameters; returned by this function
    for spec_line in specification_lines:
        halves = list(spec_line.split("|")) #First split each line into the value half and the index half
        parameter_details = list(halves[0].split()) #Get the parameter name and value, and unit if the parameter is not a CPE exponent.
        if len(parameter_details) == 3:
            parameter_name, parameter_value, parameter_unit = parameter_details[0][:2], parameter_details[1], parameter_details[2]
        elif len(parameter_details) == 2:
            parameter_name, parameter_value, parameter_unit = parameter_details[0][:2], parameter_details[1], ""
        else:
            raise ValueError("Incorrect parameter line.")
        parameter_value = float(parameter_value) #Convert parameter value to float
        parameter_index = int(halves[1]) #Get the index and convert it to an integer
        fit_parameters[parameter_index] = parameter_value #Update the parameter list
        for element in raw_units.diagram.list_elements(): #Now check to which element the parameter under investigation belongs
            if element.name == parameter_name: #If the parameter name is equal to an element name (can happen for R, C, L, Q, G):
                element.idx = parameter_index #Set the element index equal to the parameter index
                if element.tag == "R":
                    element.R = parameter_value #Set the element value equal to the parameter value
                if element.tag == "C":
                    element.C = parameter_value
                if element.tag == "L":
                    element.L = parameter_value
                if element.tag == "Q":
                    element.Q = parameter_value
                if element.tag == "O":
                    element.O = parameter_value
                if element.tag == "S":
                    element.S = parameter_value
                if element.tag == "G":
                    element.G = parameter_value
            if element.tag == "Q" and parameter_name[0] == "n" and int(parameter_name[1]) == element.number: #If the parameter is a CPE exponent
                element.idx2 = parameter_index #Set the secondary index equal to the parameter index
                element.n = parameter_value #Set the value
            if element.tag in ["O", "S", "G"] and parameter_name[0] in ["k", "l", "m"] and int(parameter_name[1]) == element.number: #If the parameter is a CPE exponent
                element.idx2 = parameter_index #Set the secondary index equal to the parameter index
                if element.tag == "O":
                    element.k = parameter_value
                if element.tag == "S":
                    element.l = parameter_value
                if element.tag == "G":
                    element.m = parameter_value
    return fit_parameters

def parseResult(filename):
    """Parse a result (.recm2) file.
    
    Arguments:
    filename -- Path to and name of the .recm2 file to be parsed
    
    Returns tuple of:
    circuit_input -- ecm_circuits.Circuit object
    dataSet object containing measured data
    list of model parameters
    """
    measured_data = [[], [], []] #Measured data, raw.
    #Booleans to describe where in the process of file reading the program currently is.
    reading_circuit_string = False
    reading_specification_lines = False
    reading_statistics = False
    reading_measured_data = False
    spec_lines = [] #Specification lines
    file = open(filename, "r") #Open a new RECM2 file
    for line in file: #Read the file line by line. Reading is done looking for headers in REVERSE ORDER. This is important.
        if line == ">IMPEDANCE FIT\n": #This part is not relevant for opening RECM2 files
            reading_measured_data = False #Set reading state
            break
        if line == ">IMPEDANCE DATA\n":
            reading_statistics = False #Set reading state
            reading_measured_data = True #Set reading state
        if reading_measured_data and line not in ["IMPEDANCE DATA\n", "Frequency (Hz), Re(Z) / Ohm, Im(Z) / Ohm\n", "\n"]: #Skip the announcement, header and empty lines
            datum = list(line.split(",")) #Process the measured data
            try:
                for d in range(len(datum)):
                    measured_data[d].append(float(datum[d]))
            except:
                pass
        if line == ">STATISTICAL DATA\n": #Look for the specification lines
            reading_specification_lines = False #Set reading state
            reading_statistics = True #Set reading state
            model_parameters = assignIndicesAndValues(spec_lines, circuit_input) #Process the specification lines
        if reading_specification_lines and line not in [">MODEL PARAMETERS\n", "\n"]: #Skip the announcement line and empty lines
            spec_lines.append(line) #Add the specification line to the list of specification lines
        if line == ">MODEL PARAMETERS\n": #Look for the specification lines
            reading_specification_lines = True #Set reading state
        if reading_circuit_string and line not in [">CIRCUIT DEFINITION\n", "\n"]: #Skip the announcement line and empty lines
            circuit_input = parseCircuitString(line[:-1]) #Read the circuit string
            reading_circuit_string = False #Set reading state
        if line == ">CIRCUIT DEFINITION\n": #Look for the circuit definition
            reading_circuit_string = True #Set reading state
    file.close()
    return circuit_input, dataSet(freq = measured_data[0], real = measured_data[1], imag = measured_data[2]), model_parameters #Returns the circuit (treated), the data and the parameter list

#######################
##RESULT FILE WRITING##
#######################

def createResultFile(circuit, parameters, e_dataset, m_dataset, default_filename = ""): #To write the result file, we need 1) the circuit diagram, 2) the parameter list, 3) the experimental data, 4) the model line's points and 5) a name for the new file.
    """Create a new result file.
    
    Arguments:
    circuit -- ecm_circuits.Circuit object
    parameters -- list of fit parameters
    e_dataset -- dataSet of measured data
    m_dataset -- dataSet of model curve
    
    Keyword arguments:
    default_filename -- initial file name, will normally be changed by the user
    
    No return value. The function ends with closing the output RECM2 file."""
    #File name
    fn = fdiag.asksaveasfilename(initialfile = default_filename)
    if fn[-6:] != ".recm2":
        fn = fn + ".recm2"
    #Create the file
    outfile = open(fn, "w")
    #Circuit definition
    outfile.write(">CIRCUIT DEFINITION\n")
    outfile.write(circuit.diagram.generate_circuit_string(verbose = True) + "\n\n")
    #Model parameter section
    e_units = {"R": "Ohm", "L": "H", "C": "F", "Q": "Fs^(n-1)", "n": "", "O": "Ohm", "S": "Ohm", "G": "Ohm", "k": "s^{1/2}", "l": "s^{1/2}", "m": "s^{1/2}"}
    outfile.write(">MODEL PARAMETERS\n")
    for e in circuit.diagram.list_elements():
        outfile.write(e.name + ": " + str(parameters[e.idx]) + " " + e_units[e.tag] + " | " + str(e.idx) + "\n")
        if e.tag == "Q":
            outfile.write("n" + str(e.number) + ": " + str(parameters[e.idx2]) + " " + e_units["n"] + " | " + str(e.idx2) + "\n")
        if e.tag == "O":
            outfile.write("k" + str(e.number) + ": " + str(parameters[e.idx2]) + " " + e_units["k"] + " | " + str(e.idx2) + "\n")
        if e.tag == "S":
            outfile.write("l" + str(e.number) + ": " + str(parameters[e.idx2]) + " " + e_units["l"] + " | " + str(e.idx2) + "\n")
        if e.tag == "G":
            outfile.write("," + str(e.number) + ": " + str(parameters[e.idx2]) + " " + e_units["m"] + " | " + str(e.idx2) + "\n")
    outfile.write("\n")
    #Statistics section
    outfile.write(">STATISTICAL DATA\n")
    parct = 0
    for c in circuit.diagram.generate_circuit_string(verbose = True):
        if c in "RCL":
            parct += 1
        elif c in "QSOG":
            parct += 2
    outfile.write("Number of parameters: " + str(parct) + "\n")
    outfile.write("Number of frequencies N_f: " + str(len(e_dataset.freq)) + "\n")
    dof, opr = 2*len(e_dataset.freq) - parct, 2*len(e_dataset.freq)/parct
    outfile.write("Degrees of freedom: " + str(dof) + "\n")
    outfile.write("Observation-to-parameter ratio: {:g}\n".format(opr))
    redsumsq = (1/dof)*sum(((np.real(circuit.impedance(parameters, e_dataset.freq)) - e_dataset.real)/e_dataset.real)**2 + ((np.imag(circuit.impedance(parameters, e_dataset.freq)) - e_dataset.imag)/e_dataset.imag)**2)
    outfile.write("Proportionally weighted reduced sum of squares S_v: {:g}\n".format(redsumsq))
    #Data section
    outfile.write("\n>IMPEDANCE DATA\n")
    outfile.write("Frequency (Hz), Re(Z) / Ohm, Im(Z) / Ohm\n")
    for d in range(len(e_dataset.freq)):
        outfile.write("{:g}, {:g}, {:g}\n".format(e_dataset.freq[d], e_dataset.real[d], e_dataset.imag[d]))
    #Model curve section
    outfile.write("\n>IMPEDANCE FIT\n")
    outfile.write("Frequency (Hz), Re(Z) / Ohm, Im(Z) / Ohm\n")
    for d in range(len(m_dataset.freq[:-1])):
        outfile.write("{:g}, {:g}, {:g}\n".format(m_dataset.freq[d], m_dataset.real[d], m_dataset.imag[d]))
    outfile.write("{:g}, {:g}, {:g}".format(m_dataset.freq[-1], m_dataset.real[-1], m_dataset.imag[-1]))
    outfile.close()

################################
##CIRCUIT PRESET FILE HANDLING##
################################

def parseCircuitPresets(filename):
    """Parse the ecm_presets.decim_circuits file.
    
    Arguments:
    filename -- name of the presets file, should be ecm_presets.decim_circuits
    
    Returns:
    valid_circuit_definitions -- list of all correct circuit strings extracted from the file"""
    file = open(filename, "r")
    valid_circuit_definitions = []
    for line in file:
        charcounts = {"(": 0, ")": 0, "{": 0, "}": 0}
        ctext = line.rstrip("\n")
        for c in ctext:
            if c in charcounts:
                charcounts[c] += 1
        #Check if any units have been defined
        if "(" not in ctext and ")" not in ctext and "{" not in ctext and "}" not in ctext:
            continue
        #Check if all units have a beginning and an end
        if charcounts["("] != charcounts[")"] or charcounts["{"] != charcounts["}"]:
            continue
        #Check if at least one element has been defined
        if "R" not in ctext and "C" not in ctext and "L" not in ctext and "Q" not in ctext and "O" not in ctext and "S" not in ctext and "G" not in ctext:
            continue
        #If the iteration has not been skipped at this point, the circuit string is valid -- or at least interpretable
        valid_circuit_definitions.append(ctext)
    file.close()
    return valid_circuit_definitions