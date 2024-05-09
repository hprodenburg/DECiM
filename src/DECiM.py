"""DECiM (Determination of Equivalent Circuit Models) is an equivalent circuit model fitting program for impedance data. It is a GUI-based program.
Much of the source code is spread over other python source files, all of which must be in the same folder as DECiM.py to ensure that the program works correctly.
DECiM was written and is maintained by Henrik Rodenburg. Current version: 1.2.17, 9 May 2024.

This is the core module -- when launched, DECiM starts. This module also defines the Window class."""

############################################################
##IMPORTS FROM OTHER MODULES AND MATPLOTLIB INITIALISATION##
############################################################

#Matplotlib (plotting)
import matplotlib as mp
import matplotlib.pyplot as pt
import matplotlib.backends.backend_tkagg as btk
import matplotlib.figure as fg
import matplotlib.animation as anim

#NumPy (calculations)
import numpy as np

#Scipy (optimization)
import scipy.optimize as op

#Copy (history)
import copy

#Functools (preset circuits)
from functools import partial

#Web browser (help)
import webbrowser

#OS (help)
import os

#Sys (font sizes)
import sys

#Tkinter (GUI)
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.simpledialog as sdiag
import tkinter.filedialog as fdiag

#Choose Matplotlib backend
mp.use("TKAgg")

#############################
##IMPORTS SPECIFIC TO DECiM##
#############################

#These modules are for the most part imported in full because technically, DECiM could have been just one file -- the only reason these are separate modules is for readability. However, the 'from module import *' statement is avoided to ensure that it is clear where all the classes and functions are coming from.
from ecm_circuits import Circuit, CircuitManager, CircuitDefinitionWindow, Circuit, Unit, Resistor, ConstantPhaseElement #Circuit models, elements, impedance calculations, and circuit selection.
from ecm_helpers import nearest, maxima #Helper functions nearest(a, b) and maxima(b).
from ecm_file_io import parseData, parseCircuitString, parseResult, createResultFile, parseCircuitPresets, DataSpecificationWindow #Functions related to parsing data files, creating result files and parsing result files.
from ecm_datastructure import dataSet #dataSet class.
from ecm_plot import PlotFrame, limiter, GeometryWindow #PlotFrame, limiter and GeometryWindow classes. DECiM's plots are plotted in a PlotFrame.
from ecm_fit import MultistartEngine, SimpleRefinementEngine, RefinementWindow #The classes dealing with the fitting procedures.
from ecm_user_input import InteractionFrame #Frame containing the various input fields, sliders, dropdowns, etc. that make up the interactive part of the program.
from ecm_history import expandedDataSet, HistoryManager, DataSetSelectorWindow #For non-interactive plotting and quick switching between datasets.
from ecm_manual import HelpWindow #User instructions.
from ecm_zhit import ZHITWindow #For data validation.
import ecm_custom_models as ecmcm #For custom models.

#########################
##BASIC PLOTTING LAYOUT##
#########################

font_sizes = {"small": 12, "medium": 16, "large": 18}
if len(sys.argv) > 1:
    if sys.argv[1] == "-b":
        font_sizes = {"small": 22, "medium": 26, "large": 28}

#Taken from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
pt.rc('font', size = font_sizes["medium"])
pt.rc('axes', titlesize = font_sizes["large"])
pt.rc('axes', labelsize = font_sizes["medium"])
pt.rc('xtick', labelsize = font_sizes["small"])
pt.rc('ytick', labelsize = font_sizes["small"])
pt.rc('legend', fontsize = font_sizes["medium"])
pt.rc('figure', titlesize = font_sizes["large"])

#############
##DECiM GUI##
#############

#Useful links:
# with the help of https://zetcode.com/tkinter/ by Jan Bodnar
# https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_tk_sgskip.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_subplot
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twinx.html
# Matplotlib pages on Figures & Axes
#o https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib

class Window(ttk.Frame):
    def __init__(self, width, height, fontsize = ""):
        """Core class of DECiM. From here, all other parts of the program are accessed.
        
        Init arguments:
        width -- Window width
        height -- Window height
        
        Attributes:
        width -- Window width
        height -- Window height
        circuit_manager -- ecm_circuits.CircuitManager object
        history_manager -- ecm_history.HistoryManager object
        data -- dataSet containing measured data
        model -- dataSet containing model curve
        prevparams -- list of lists of parameters, used to save old parameter sets that can be returned to with the Undo Refinement option in the Calculate menu
        fitlim -- Boolean indicating whether or not the simple refinement should have frequency limits
        fitfreq -- frequency limits for the simple refinement
        master -- Tkinter tk that is actually running the show
        plots -- ecm_plots.PlotFrame containing the plots
        interactive -- ecm_user_input.InteractionFrame containing the controls below the plots
        
        Methods:
        make_UI -- make the UI
        
        make_menus -- make the menus
        For making the menus: make_calculatemenu, make_circuitmenu, make_filemenu, make_helpmenu, make_historymenu and make_plotmenu
        Circuit presets: addCircuitPresets, addCustomModels
        
        make_circuitbutton -- open the circuit drawing window
        on_screen -- properly scale the window and finish building the UI
        
        For file handling: loadData, loadResult, saveResult
        
        For plot layout: setVisRY1, setVisRY2, setLogRY1, setLogRY2, setABAdmittance, setBodeZPhi, setComplexPlaneY, setComplexPlaneZ, setABImpedance, setABSigma, setABEpsilon
        
        resetView -- reset view limits of plots
        generateFit -- calculate the model curve
        canvasUpdate -- update limits, calculate model curve, update view
        
        setSampleGeometry -- set sample thickness and area
        
        Saving and loading data from measurements (RAM): selectOtherDataset, plotNonInteractive, saveToHistory
        
        Refinement: minRefine, advancedRefinement, setRefinementRange, undoRefinement
        
        performZHIT -- launch Z-HIT transform window
        
        Circuit selection: setCircuit, selectCircuit, setCustomModel, typeCircuit
        
        showInstructions -- launch the Help window
        
        markFrequencies -- mark frequencies that are integer powers of 10"""
        super().__init__()
        #Frame width and height
        self.width = width
        self.height = height
        #Font size
        self.fontsize = fontsize
        #Defining a circuit
        self.circuit_manager = CircuitManager()
        #Saving of datasets in memory
        self.history_manager = HistoryManager()
        #Creating the data and model datastructures
        self.data = dataSet()
        self.model = dataSet()
        #Previous parameters
        self.prevparams = []
        #Parameters regarding limiting the scope of the fit
        self.fitlim = False
        self.fitfreq = None
        #Finish building the UI
        self.make_UI()

    #UI INITIALISATION FUNCTION

    def make_UI(self):
        """Create the DECiM core window UI: the menu bar, plots, slider, dowpdown, buttons, and entry boxes."""
        self.master.title("DECiM")
        self.plots = PlotFrame((self.width, self.height), self.data, self.history_manager.dataset_dict, self.history_manager.visible)
        self.plots.pack(fill = tk.BOTH, expand = tk.YES)
        self.interactive = InteractionFrame(self.circuit_manager.circuit, list(np.zeros(100) + 1), self.canvasUpdate)
        self.interactive.pack(fill = tk.BOTH, expand = tk.YES)
        self.prevparams.append(self.interactive.parameters)
        self.pack(fill = "both", expand = True)
        self.make_menus()
        self.make_circuitbutton()
        self.on_screen()

    #MENU INITIALISATION FUNCTIONS

    def make_menus(self):
        """Create the menu bar. The tabs of the menu are created in separate functions."""
        menubar = tk.Menu(self.master)
        self.master.config(menu = menubar)
        self.make_filemenu(menubar)
        self.make_calculatemenu(menubar)
        self.make_plotmenu(menubar)
        self.make_circuitmenu(menubar)
        self.make_historymenu(menubar)
        self.make_helpmenu(menubar)

    def make_filemenu(self, in_menu):
        """Create the File menu in the menu bar."""
        fileMenu = tk.Menu(in_menu)
        fileMenu.add_command(label = "Load data...", command = self.loadData)
        fileMenu.add_command(label = "Load result...", command = self.loadResult)
        fileMenu.add_separator()
        fileMenu.add_command(label = "Save result...", command = self.saveResult)
        fileMenu.add_separator()
        fileMenu.add_command(label = "Specify data file layout...", command = self.defineDataFile)
        fileMenu.add_separator()
        fileMenu.add_command(label = "Exit", command = self.exitApplication)
        in_menu.add_cascade(label = "File", menu = fileMenu)

    def make_plotmenu(self, in_menu):
        """Create the Plot menu in the menu bar."""
        plotMenu = tk.Menu(in_menu)
        plotMenu.add_command(label = "Toggle data visibility", command = self.toggleDataVisibility)
        plotMenu.add_command(label = "Toggle fit visibility", command = self.toggleFitVisibility)
        plotMenu.add_command(label = "Mark frequencies that are integer powers of 10", command = self.markFrequencies)
        plotMenu.add_separator()
        plotMenu.add_command(label = "Reset view", command = self.resetView)
        plotMenu.add_separator()
        plotMenu.add_command(label = "Set sample geometry", command = self.setSampleGeometry)
        plotMenu.add_separator()
        plotMenu.add_command(label = "Toggle RHS primary log-scale", command = self.setLogRY1)
        plotMenu.add_command(label = "Toggle RHS secondary log-scale", command = self.setLogRY2)
        plotMenu.add_command(label = "Toggle RHS primary visibility", command = self.setVisRY1)
        plotMenu.add_command(label = "Toggle RHS secondary visibility", command = self.setVisRY2)
        plotMenu.add_separator()
        plotMenu.add_command(label = "Complex plane impedance", command = self.setComplexPlaneZ)
        plotMenu.add_command(label = "Complex plane admittance", command = self.setComplexPlaneY)
        plotMenu.add_command(label = "Bode amplitude/phase", command = self.setBodeZPhi)
        plotMenu.add_command(label = "Y\' and Y\'\' vs. frequency", command = self.setABAdmittance)
        plotMenu.add_command(label = "Z\' and Z\'\' vs. frequency", command = self.setABImpedance)
        plotMenu.add_command(label = "Conductivity vs. frequency", command = self.setABSigma)
        plotMenu.add_command(label = "Permittivity vs. frequency", command = self.setABEpsilon)
        in_menu.add_cascade(label = "Plot", menu = plotMenu)

    def make_calculatemenu(self, in_menu):
        """Create the Calculate menu in the menu bar."""
        calculateMenu = tk.Menu(in_menu)
        calculateMenu.add_command(label = "Automatic initial guess", command = self.automaticInitialGuess)
        calculateMenu.add_command(label = "Refine solution (simple)", command = self.minRefine)
        calculateMenu.add_command(label = "Set simple refinement frequency range", command = self.setRefinementRange)
        calculateMenu.add_command(label = "Advanced refinement...", command = self.advancedRefinement)
        calculateMenu.add_command(label = "Undo refinement", command = self.undoRefinement)
        calculateMenu.add_separator()
        calculateMenu.add_command(label = "Get apex frequencies", command = self.getMaxima)
        calculateMenu.add_separator()
        calculateMenu.add_command(label = "Perform Z-HIT transform", command = self.performZHIT)
        in_menu.add_cascade(label = "Calculate", menu = calculateMenu)
        
    def make_circuitmenu(self, in_menu):
        """Create the Circuit menu in the menu bar."""
        circuitMenu = tk.Menu(in_menu)
        circuitMenu.add_command(label = "Draw circuit", command = self.selectCircuit)
        circuitMenu.add_command(label = "Type circuit", command = self.typeCircuit)
        circuitMenu.add_separator()
        self.addCircuitPresets(circuitMenu)
        circuitMenu.add_separator()
        self.addCustomModels(circuitMenu)
        in_menu.add_cascade(label = "Circuit", menu = circuitMenu)

    def make_historymenu(self, in_menu):
        """Create the History menu in the menu bar."""
        historyMenu = tk.Menu(in_menu)
        historyMenu.add_command(label = "Save current dataset to history", command = self.saveToHistory)
        historyMenu.add_command(label = "Select other dataset", command = self.selectOtherDataset)
        historyMenu.add_separator()
        historyMenu.add_command(label = "Plot or remove non-interactive dataset (max. 3)", command = self.plotNonInteractive)
        in_menu.add_cascade(label = "History", menu = historyMenu)
    
    def make_helpmenu(self, in_menu):
        """Create the Help menu in the menu bar."""
        helpMenu = tk.Menu(in_menu)
        helpMenu.add_command(label = "Show instructions", command = self.showInstructions)
        helpMenu.add_command(label = "Open manual", command = self.openManual)
        in_menu.add_cascade(label = "Help", menu = helpMenu)

    #OTHER UI ELEMENT INITIALISATION FUNCTIONS

    def make_circuitbutton(self):
        """Create the button which launches the circuit drawing window."""
        self.circop = tk.StringVar(self)
        self.circop.set(self.circuit_manager.circuit.diagram.generate_circuit_string(verbose = True))
        self.selector_launcher = tk.Button(self.interactive.slider_frame, textvariable = self.circop, command = self.selectCircuit)
        self.selector_launcher.pack(side = tk.BOTTOM, anchor = tk.S)

    def on_screen(self):
        """Scale the window to the dimensionms of the screen and finish the UI initialization."""
        wd, ht = self.width, self.height
        sw, sh = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        x, y = int((sw - wd)/2), int((sh - ht)/2)
        self.master.geometry("{:d}x{:d}+{:d}+{:d}".format(wd, ht, x, y))

    #EVENT DRIVEN FUNCTIONS

    #Loading and saving files

    def defineDataFile(self):
        """Open a DataSpecificationWindow to specify what information is contained in which column of the datafiles."""
        DataSpecificationWindow("ecm_datafiles.decim_specification")

    def loadData(self):
        """Open a dialog window for opening data (.txt) files. Import the data, set the plot title and circuit, and update the plots with the new data."""
        fn = fdiag.askopenfilename(filetypes = [("Text files", ".txt"), ("All files", "*.*")])
        self.data = parseData(fn)
        try:
            self.plots.title = list(fn.split("/"))[-1]
        except:
            self.plots.title = fn
        if len(self.circuit_manager.circuit.diagram.list_elements()) == 0: #Make sure there is a valid circuit to start the fit with if it hasn't been made yet.
            self.circuit_manager.circuit = Circuit()
            self.circuit_manager.circuit.diagram.add_element(Resistor(1, 0, 0, self.circuit_manager.circuit.diagram))
            self.circuit_manager.circuit.diagram.add_element(ConstantPhaseElement(1, 1, 0, 1, self.circuit_manager.circuit.diagram))
            self.interactive.circuit = self.circuit_manager.circuit
            self.circop.set(self.circuit_manager.circuit.diagram.generate_circuit_string(verbose = True))
        self.canvasUpdate()
        self.resetView()

    def loadResult(self):
        """Open a dialog window for opening result (.recm2) files. Import the data and model, set the plot title and circuit, and update the plots with the new data and model."""
        fn = fdiag.askopenfilename(filetypes = [("Result files", ".recm2"), ("All files", "*.*")])
        result = parseResult(fn)
        circmodel, self.data, self.interactive.parameters = result[:3]
        if result[3]: #Custom model
            ecmcm.override_impedance_method = True
            self.setCustomModel(circmodel, result[4])
        else: #No custom model
            self.setCircuit(circmodel)
        try:
            self.plots.title = list(fn.split("/"))[-1]
        except:
            self.plots.title = fn
        self.canvasUpdate()
        self.resetView()

    def saveResult(self):
        """Open a dialog window for saving result (.recm2) files."""
        createResultFile(self.circuit_manager.circuit, self.interactive.parameters, self.data, self.model, default_filename = self.plots.title.rstrip(".txt"))

    #Exiting DECiM
    
    def exitApplication(self):
        """Exit DECiM."""
        self.quit()
        self.destroy()

    #Plot layout
    
    def setLogRY1(self):
        """Toggle log scale for the RHS primary y axis."""
        self.plots.logRY1 = not self.plots.logRY1
        self.resetView()

    def setLogRY2(self):
        """Toggle log scale for the RHS secondary y axis."""
        self.plots.logRY2 = not self.plots.logRY2
        self.resetView()
        
    def setVisRY1(self):
        """Toggle visibility of the amplitude or real admittance."""
        self.plots.visRY1 = not self.plots.visRY1
        self.canvasUpdate()
        
    def setVisRY2(self):
        """Toggle visibility of the phase or imaginary admittance."""
        self.plots.visRY2 = not self.plots.visRY2
        self.canvasUpdate()
    
    def setABAdmittance(self):
        """Switch to the real+imaginary admittance plot."""
        if self.plots.rhs_type != "YY vs. f":
            self.plots.prev_rhs_type = self.plots.rhs_type
            self.plots.rhs_type = "YY vs. f"
            if not self.plots.logRY1:
                self.plots.logRY1 = True
            if not self.plots.logRY2:
                self.plots.logRY2 = True
            self.resetView()
            
    def setABImpedance(self):
        """Switch to the real+imaginary admittance plot."""
        if self.plots.rhs_type != "ZZ vs. f":
            self.plots.prev_rhs_type = self.plots.rhs_type
            self.plots.rhs_type = "ZZ vs. f"
            if not self.plots.logRY1:
                self.plots.logRY1 = True
            if not self.plots.logRY2:
                self.plots.logRY2 = True
            self.resetView()
        
    def setBodeZPhi(self):
        """Switch to the Bode amplitude/phase plot."""
        if self.plots.rhs_type != "Bode amplitude/phase":
            self.plots.prev_rhs_type = self.plots.rhs_type
            self.plots.rhs_type = "Bode amplitude/phase"
            if not self.plots.logRY1:
                self.plots.logRY1 = True
            if self.plots.logRY2:
                self.plots.logRY2 = False
            self.resetView()
            
    def setABSigma(self):
        """Switch to the real+imaginary conductivity plot."""
        if self.plots.rhs_type != "sigma vs. f":
            self.plots.prev_rhs_type = self.plots.rhs_type
            self.plots.rhs_type = "sigma vs. f"
            if not self.plots.logRY1:
                self.plots.logRY1 = True
            if not self.plots.logRY2:
                self.plots.logRY2 = True
            self.resetView()

    def setABEpsilon(self):
        """Switch to the real+imaginary permittivity plot."""
        if self.plots.rhs_type != "epsilon vs. f":
            self.plots.prev_rhs_type = self.plots.rhs_type
            self.plots.rhs_type = "epsilon vs. f"
            if not self.plots.logRY1:
                self.plots.logRY1 = True
            if not self.plots.logRY2:
                self.plots.logRY2 = True
            self.resetView()
        
    def setComplexPlaneY(self):
        """Switch to the complex plane admittance plot."""
        if self.plots.lhs_type != "Complex plane Y":
            self.plots.prev_lhs_type = self.plots.lhs_type
            self.plots.lhs_type = "Complex plane Y"
            self.resetView()

    def setComplexPlaneZ(self):
        """Switch to the complex plane impedance plot."""
        if self.plots.lhs_type != "Complex plane Z":
            self.plots.prev_lhs_type = self.plots.lhs_type
            self.plots.lhs_type = "Complex plane Z"
            self.resetView()
    
    def toggleDataVisibility(self):
        """Toggle visibility of the data."""
        self.plots.datavis = not self.plots.datavis
        self.canvasUpdate()
    
    def toggleFitVisibility(self):
        """Toggle visibility of the fit."""
        self.plots.fitvis = not self.plots.fitvis
        self.canvasUpdate()

    def resetView(self):
        """Reset the plots' view limits."""
        self.plots.limiter.enabled = False
        self.canvasUpdate()
        self.plots.limiter.enabled = True
        
    def setSampleGeometry(self):
        """Set the sample thickness and area, then reset the view."""
        geometry_window = GeometryWindow(self.plots)
        self.resetView()

    #History functions

    def selectOtherDataset(self):
        """Launch the dataset selector window and replace the current data and model with the selected model."""
        selector_window = DataSetSelectorWindow(self.history_manager)
        selector_window.wait_window()
        new_dataset = self.history_manager.dataset_dict[selector_window.chosen_dataset.get()]
        self.interactive.parameters = new_dataset.parameters
        self.data = new_dataset.data
        self.model = new_dataset.model
        self.circuit_manager.circuit.diagram = new_dataset.circuit.diagram
        self.plots.title = new_dataset.name
        self.canvasUpdate()

    def plotNonInteractive(self):
        """Launch the dataset selector window and add or remove the non-interactive selected data and model."""
        selector_window = DataSetSelectorWindow(self.history_manager)
        selector_window.wait_window()
        self.history_manager.toggle_dataset_visibility(selector_window.chosen_dataset.get())
        self.canvasUpdate()

    def saveToHistory(self):
        """Save the current data and model to History, where they can be accessed later."""
        full_dataset = expandedDataSet(copy.copy(self.plots.title), copy.copy(self.data), copy.copy(self.model), copy.copy(self.circuit_manager.circuit), parameters = copy.copy(self.interactive.parameters))
        self.history_manager.add_dataset(full_dataset)

    #Fitting functions
    
    def automaticInitialGuess(self):
        """Create an automatic initial guess using a MultistartEngine."""
        #Indicate that the initial guess is being calculated
        self.plots.lhs.text(0.7, 0.9, "Guessing...", transform = self.plots.lhs.transAxes)
        self.plots.canvas.draw()
        self.update()
        #Create a dictionary of the parameters (as in ecm_fit.RefinementWindow, but opposite)
        par_dict = {}
        for p in self.circuit_manager.circuit.diagram.list_elements(): #Tie parameter names to parameter array indices and count the parameters
            par_dict[p.idx] = p.name
            if p.tag in "QSOGH":
                if p.tag == "Q":
                    par_dict[p.idx2] = "n" + str(p.number)
                if p.tag == "O":
                    par_dict[p.idx2] = "k" + str(p.number)
                if p.tag == "S":
                    par_dict[p.idx2] = "l" + str(p.number)
                if p.tag == "G":
                    par_dict[p.idx2] = "m" + str(p.number)
                if p.tag == "H":
                    par_dict[p.idx2] = "t" + str(p.number)
                    par_dict[p.idx3] = "b" + str(p.number)
                    par_dict[p.idx4] = "g" + str(p.number)
        #Calculate the initial guess and update previous parameters
        guess_engine = MultistartEngine(self.circuit_manager.circuit.impedance, self.circuit_manager.circuit.jnp_impedance, self.data, list(np.ones(len(par_dict))), par_dict, self.prevparams, opt_module = "SciPy", opt_method = "Nelder-Mead", silent = True, nmaxiter = 100, weighting_scheme = "Unit", starts_per_par = 3, nmaxstarts = 30)
        guess_engine.generate_solution()
        #Update current parameters and the canvas
        self.prevparams.append(self.interactive.parameters)
        for p in par_dict:
            self.interactive.parameters[p] = guess_engine.parameters[p]
        self.canvasUpdate()

    def minRefine(self):
        """Optimize all model parameters using a unit weighting scheme and update the plots."""
        self.plots.lhs.text(0.7, 0.9, "Refining...", transform = self.plots.lhs.transAxes)
        self.plots.canvas.draw()
        self.update()
        if self.fitlim:
            optim_data = dataSet(freq = self.data.freq[self.fitfreq[1]:self.fitfreq[0]], real = self.data.real[self.fitfreq[1]:self.fitfreq[0]], imag = self.data.imag[self.fitfreq[1]:self.fitfreq[0]])
        else:
            optim_data = self.data
        refinement_engine = SimpleRefinementEngine(self.interactive.parameters, optim_data, self.circuit_manager.circuit.impedance)
        refinement_engine.minRefinement()
        self.prevparams.append(self.interactive.parameters) #Save the previous parameters to allow the refinement to be undone.
        self.interactive.parameters = refinement_engine.output_params
        self.canvasUpdate()
    
    def setRefinementRange(self):
        """Open a simple dialog box to set the frequency limits used in the minRefine function."""
        self.fitlim = True
        sset = sdiag.askstring(title = "Provide frequency range for fitting", prompt = "Highest, lowest frequency (in Hz, comma separated):")
        hlim, llim = sset.split(",")
        hlim, llim = float(hlim), float(llim)
        hidx, lidx = nearest(hlim, self.data.freq), nearest(llim, self.data.freq)
        self.fitfreq = (hidx,lidx)
    
    def advancedRefinement(self):
        """Launch the refinement window and wait for the result. If the user accepts the result, replace the model parameters with the result and update the plots."""
        refinement_window = RefinementWindow(self.interactive.circuit.impedance, self.interactive.circuit.jnp_impedance, self.interactive.circuit.diagram.list_elements(), self.interactive.parameters, self.data)
        refinement_window.wait_window()
        if refinement_window.refinement_accepted:
            self.prevparams.append(self.interactive.parameters) #Save the previous parameters to allow the refinement to be undone.
            self.interactive.parameters = refinement_window.refined_parameters
        self.canvasUpdate()

    def undoRefinement(self):
        """Replace the current model parameters with the previous parameters and remove the newest entry in the list of previous parameters."""
        self.interactive.parameters = self.prevparams[-1]
        del self.prevparams[-1]
        self.canvasUpdate()
        
    #Z-HIT data validation
    
    def performZHIT(self):
        """Launch a Z-HIT window and wait for the result. If the result is accepted, replace the data with the Z-HIT fit result."""
        zhit_window = ZHITWindow(self.data)
        zhit_window.wait_window()
        if zhit_window.correction_accepted:
            self.data = zhit_window.data
            self.plots.title += " | Z-HIT"
        self.canvasUpdate()

    #Update function. Extremely important and commonly used.
    
    def canvasUpdate(self):
        """General plot update function. Update all limits, generate the model line from the parameters and the circuit impedance function, and update the plots."""
        amlims = self.plots.right_plot.primary.viewLim.get_points()
        comlims = self.plots.left_plot.primary.viewLim.get_points()
        phalims = self.plots.right_plot.twin.viewLim.get_points()
        self.plots.limiter.freq = (amlims[0][0], amlims[1][0])
        self.plots.limiter.real = (comlims[0][0], comlims[1][0])
        self.plots.limiter.imag = (comlims[0][1], comlims[1][1])
        self.plots.limiter.amp = (amlims[0][1], amlims[1][1])
        self.plots.limiter.phase = (phalims[0][1], phalims[1][1])
        self.generateFit()
        self.plots.updatePlots(self.data, self.history_manager.dataset_dict, self.history_manager.visible, model = self.model)

    #Changing the circuit
    
    def setCircuit(self, circuit_model):
        """Change the circuit model to a new circuit whose impedance function matches the expected function from the circuit diagram.
        
        Arguments:
        self -- needs access to instance of Window class, like all other functions defined in it
        circuit_model -- a Circuit object from ecm_circuits"""
        ecmcm.override_impedance_method = False
        self.circuit_manager.circuit = circuit_model
        self.interactive.circuit = circuit_model
        self.circop.set(circuit_model.diagram.generate_circuit_string(verbose = True))
        self.interactive.reset_parameter_listbox()
        
    def setCustomModel(self, circuit_model, impedance_function_name):
        """Change the circuit model to a new circuit with a custom impedance function.
        
        Arguments:
        self -- needs access to instance of Window class, like all other functions defined in it
        circuit_model -- a Circuit object from ecm_circuits
        impedance_function_name -- any key in the custom_model_diagrams dictionary in ecm_custom_models
        circuit_name -- string containing the name of the custom model"""
        ecmcm.override_impedance_method = True
        ecmcm.custom_model_name = impedance_function_name
        self.circuit_manager.circuit = circuit_model
        self.interactive.circuit = circuit_model
        self.circop.set(impedance_function_name)
        self.interactive.reset_parameter_listbox()

    def selectCircuit(self):
        """Open a circuit drawing window and wait for it to be closed to obtain a new circuit. Then set the model circuit to the obtained circuit."""
        selector_window = CircuitDefinitionWindow(self.circuit_manager.circuit)
        selector_window.wait_window()
        self.setCircuit(selector_window.chosen_circuit)
        
    def typeCircuit(self):
        """Open a simple dialog box in which a circuit string can be typed. Generate a circuit from the string and set the model circuit to this new circuit."""
        cset = sdiag.askstring(title = "Circuit selection", prompt = "Circuit code:")
        try:
            self.setCircuit(parseCircuitString(cset))
        except:
            self.setCircuit(parseCircuitString("(R0Q0)"))

    def addCircuitPresets(self, submenu):
        """Create the circuit presets that use the normal impedance function as defined in ecm_presets.decim_circuits."""
        presets = parseCircuitPresets("ecm_presets.decim_circuits")
        for p in presets:
            submenu.add_command(label = p, command = partial(self.setCircuit, parseCircuitString(p)))
            
    def addCustomModels(self, submenu):
        """Create the circuit presets defined in ecm_custom_models."""
        for m in ecmcm.custom_model_diagrams:
            submenu.add_command(label = m, command = partial(self.setCustomModel, parseCircuitString(ecmcm.custom_model_diagrams[m][0]), m))

    #HELPER FUNCTIONS
            
    def showInstructions(self):
        """Launch the help window."""
        HelpWindow()
        
    def openManual(self):
        """Open the PDF manual."""
        manual_path = os.path.realpath("../Manual.pdf")
        webbrowser.open_new(r'file://' + manual_path) #See https://stackoverflow.com/questions/19453338/opening-pdf-file

    def generateFit(self):
        """Generate the model dataset from the model parameters, circuit model and a logarithmically spaced NumPy array of frequencies.
        Alternatively, update the error label in case the model generation fails."""
        xdt = 10**np.linspace(np.log10(min(self.data.freq)), np.log10(max(self.data.freq)), 500)
        self.interactive.error_label.configure(state = "disabled")
        np.seterr(all = "raise")
        try:
            imp = xdt, self.interactive.circuit.impedance(self.interactive.parameters, xdt)
            self.model.freq = xdt
            self.model.real = np.real(imp[1])
            self.model.imag = np.imag(imp[1])
            self.model.phase = np.angle(imp[1])
            self.model.amplitude = np.absolute(imp[1])
        except FloatingPointError:
            self.interactive.error_label.configure(state = "normal")
        np.seterr(all = "print")

    def getMaxima(self):
        """Using the model curve, get the maxima of the complex plane plot, then launch a window with a Text widget in which the frequencies at the maxima are displayed."""
        max_indices = maxima(-self.model.imag)
        max_points = "Apex frequencies based on MODEL\n(Angular frequency, Re[Z], -Im[Z]):\n\n"
        for i in max_indices:
            max_points += ("{:g} Hz, {:g} Ohm, {:g} Ohm\n".format(2*np.pi*self.model.freq[i], self.model.real[i], -self.model.imag[i]))
        resultbox = tk.Toplevel(self)
        resultbox.geometry("300x200")
        resulttext = tk.Text(resultbox)
        resulttext.pack()
        resulttext.insert(tk.END, max_points)

    def markFrequencies(self):
        """Place or remove circles, lines and text on the complex plane plot to indicate datapoints whose frequency is an integer power of 10. If the exact frequency is not present, the nearest point is marked."""
        #If frequencies are already being marked, stop doing so.
        if len(self.plots.mfreq_real) > 0:
            self.plots.mfreq_freq = np.array([])
            self.plots.mfreq_real = np.array([])
            self.plots.mfreq_imag = np.array([])
            self.canvasUpdate()
            return
        #Add frequencies to be marked to the plot frame.
        self.plots.mfreq_real = []
        self.plots.mfreq_imag = []
        self.plots.mfreq_freq = []
        scan_freqs = list(range(-6, 13, 1))
        new_scan_freqs = []
        for s in scan_freqs:
            if s <= max(np.log10(self.data.freq)) and s >= min(np.log10(self.data.freq)):
                new_scan_freqs.append(s)
        for integer_frequency in new_scan_freqs:
            f_idx = nearest(integer_frequency, np.log10(self.data.freq))
            self.plots.mfreq_freq.append(10**integer_frequency)
            self.plots.mfreq_real.append(self.data.real[f_idx])
            self.plots.mfreq_imag.append(self.data.imag[f_idx])
        self.plots.mfreq_freq = np.array(self.plots.mfreq_freq)
        self.plots.mfreq_real = np.array(self.plots.mfreq_real)
        self.plots.mfreq_imag = np.array(self.plots.mfreq_imag)
        self.canvasUpdate()

#############
##MAIN LOOP##
#############

root = tk.Tk()
app = Window(1280, 600)
root.mainloop()
