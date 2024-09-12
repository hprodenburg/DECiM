"""Part of DECiM. This file contains the code for the correlation matrix. Last modified 12 September 2024 by Henrik Rodenburg.

Classes:
CorrelationMatrix -- the covariance matrix from refinements divided by a matrix with elements corresponding to the product of the elements' standard deviations
CorrelationWindow -- window in which the correlation matrix is displayed"""

###########
##IMPORTS##
###########

import numpy as np

import matplotlib as mp
import matplotlib.pyplot as pt
import matplotlib.backends.backend_tkagg as btk
import matplotlib.figure as fg
import matplotlib.animation as anim

import tkinter as tk
import tkinter.ttk as ttk

############################
##CORRELATION MATRIX CLASS##
############################

class CorrelationMatrix():
    def __init__(self, covariance_matrix, standard_deviations):
        """Correlation matrix from refinements. Calculated from the covariance matrix.
        
        Init arguments:
        self
        covariance_matrix -- covariance matrix from refinement
        standard deviations -- square root of the diagonal of the covariance matrix; returned separately from refinements and therefore not calculated here but instead received as an argument
        
        Attributes:
        matrix -- NumPy array, the correlation matrix itself"""
        corr_corr = np.ones((len(standard_deviations), len(standard_deviations)))
        for i in range(len(standard_deviations)):
            for j in range(len(standard_deviations)):
                corr_corr[i, j] *= standard_deviations[i]*standard_deviations[j]
        self.matrix = covariance_matrix / corr_corr
        
############################
##CORRELATION WINDOW CLASS##
############################

class CorrelationWindow(tk.Toplevel):
    def __init__(self, correlation_matrix, parameter_dict):
        """Correlation matrix window. Builds the GUI around the CorrelationMatrix.
        
        Init arguments:
        self
        correlation_matrix -- CorrelationMatrix object
        parameter_dict -- parameter dictionary of the standard type; see ecm_fit.ParameterDictionary
        
        Attributes:
        correlation_matrix -- CorrelationMatrix object
        parameter_dict -- parameter dictionary of the standard type; see ecm_fit.ParameterDictionary
        
        width -- window width
        height -- window height
        
        plotting_frame -- ttk.Frame on which the correlation matrix is plotted
        
        Methods:
        make_UI -- make the UI
        make_plotting_frame -- make the frame on which the correlation matrix is plotted"""
        super().__init__()
        
        self.title("Correlation matrix from most recent refinement")
        
        self.width = int(self.winfo_screenwidth()*0.35)
        self.height = int(self.winfo_screenheight()*0.35)
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        
        self.correlation_matrix = correlation_matrix
        self.parameter_dict = parameter_dict
        
        self.make_UI()
        
    def make_UI(self):
        """Make the (very simple) CorrelationWindow UI."""
        self.plotting_frame = ttk.Frame(self)
        self.plotting_frame.pack(side = tk.TOP, anchor = tk.CENTER, fill = tk.BOTH, expand = tk.YES)
        self.make_plotting_frame()
        
    def make_plotting_frame(self):
        """Plot the correlation matrix on the plotting frame."""
        #Plot initialization
        self.fig, self.corr_coloured = pt.subplots(nrows = 1, ncols = 1)
        self.canvas = btk.FigureCanvasTkAgg(self.fig, self.plotting_frame)
        self.canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        #Toolbar initialization
        self.toolbar = btk.NavigationToolbar2Tk(self.canvas, self.plotting_frame)
        self.toolbar.update()
        #Plotting
        self.corr_coloured.cla()
        cax = self.corr_coloured.matshow(self.correlation_matrix, cmap = "coolwarm")
        #Setting axis ticks to parameter names
        p_len = len(self.correlation_matrix[0])
        clabels = list(np.zeros(p_len))
        for i in self.parameter_dict:
            clabels[i] = self.parameter_dict[i]
        self.corr_coloured.set_xticks(list(range(p_len)), labels = clabels)
        self.corr_coloured.set_yticks(list(range(p_len)), labels = clabels)
        #Adding a color bar
        pt.colorbar(cax, location = "right")
        #Drawing
        self.canvas.draw()