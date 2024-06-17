"""Part of DECiM. This file contains the plotting canvas class and some additional plotting-related code. Last modified 17 June 2024 by Henrik Rodenburg.

Classes:
ImpedancePlot -- base class for all plot panels; possible subclasses below:
    ComplexPlaneImpedancePlot -- Z'' vs. Z'
    ComplexPlaneAdmittancePlot -- Y'' vs. Y'
    ImpedanceFrequencyPlot -- Z' and Z'' vs. f
    AdmittanceFrequencyPlot -- Y' and Y'' vs. f
    BodePhaseAmplitudePlot -- |Z| and phi vs. f
    ConductivityFrequencyPlot -- sigma' and sigma'' vs. f
    PerimttivityFrequencyPlot -- epsilon' and epsilon'' vs. f

limiter -- handles the upper and lower limits of the plots

PlotFrame -- ttk.Frame with the plot canvas and toolbar

GeometryWindow -- window with two entries to set sample thickness and area"""

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

from ecm_datastructure import dataSet
from ecm_history import expandedDataSet

################
##SOME READING##
################

#Made with the help of https://zetcode.com/tkinter/ by Jan Bodnar
#Also https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_tk_sgskip.html
#Also https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
#Also https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_subplot
#Also https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twinx.html
#Also Matplotlib pages on Figures & Axes
#Also https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib

###########
##LIMITER##
###########

class limiter():
    def __init__(self, freq = [0.5, 1.066e6], real = [0, 1e7], imag = [0, -1e7], amp = [0.5, 2e7], phase = [-100, 100], enabled = False):
        """Defines the plot limits for all parts of the impedance. Used to fix the axis limits to just beyond the data minima and maxima, so the plot does not rescale automatically when the model is updated.
        
        Init arguments:
        freq -- list of [lower limit (Hz), upper limit (Hz)]
        real -- list of [lower limit (Ohm), upper limit (Ohm)] for the complex plane plot
        imag -- list of [upper limit (Ohm), lower limit (Ohm)] for the complex plane plot
        amp -- list of [lower limit (Ohm), upper limit (Ohm)] for the Bode plot
        phase -- list of [lower limit (degrees), upper limit (degrees)] for the Bode plot
        
        Keyword arguments:
        enabled -- indicates if limiter is being used to set plot limits; if False, the limiter is being overridden; default False. If it is enabled, the buttons in the toolbar do not work, so this should usually be False.
        
        Attributes:
        freq -- list of [lower limit (Hz), upper limit (Hz)]
        real -- list of [lower limit (Ohm), upper limit (Ohm)] for the complex plane plot
        imag -- list of [upper limit (Ohm), lower limit (Ohm)] for the complex plane plot
        amp -- list of [lower limit (Ohm), upper limit (Ohm)] for the Bode plot
        phase -- list of [lower limit (degrees), upper limit (degrees)] for the Bode plot
        enabled -- indicates if limiter is being used to set plot limits; if False, the limiter is being overridden; default False"""
        self.freq = [freq[0], freq[1]] #Lower and upper bounds of the frequency. Other quantities below.
        self.real = [real[0], real[1]]
        self.imag = [imag[0], imag[1]]
        self.amp = [amp[0], amp[1]]
        self.phase = [phase[0], phase[1]]
        self.enabled = enabled

###################
##PLOTTING PANELS##
###################

class ImpedancePlot():
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = True, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = True, primary_data_colour = "#114077", twin_data_colour = "#117740", primary_model_colour = "#4444DD", twin_model_colour = "#44DD44"):
        """Base class for all plot panels: Nyquist plots, Bode plots, etc.
        
        Init arguments becoming attributes under the same name:
        primary -- Matplotlib Axes holding the primary y axis
        data -- dataSet of experimental data
        model -- dataSet of model curve
        ghost_data -- dictionary of {'dataset name': ecm_history.expandedDataSet}; holds all data and models saved to History
        ghost_data_visibility -- list of names in ghost_data; all names in this list are of dataSets that should be visible
        
        make_twinx -- Bool; indicates if a twinx plot should be made. This is done here in __init__
        
        data_on -- Bool; display data
        model_on -- Bool; display model
        primary_axis_on -- Bool; display normal (x, y) data
        twin_axis_on -- Bool; display twinx (x, y) data
        
        Other attributes:
        ghost_colours -- colours for ghost data
        ghost_m_colours -- colours for ghost models
        
        twin -- Axes, generated via self.twinx
        
        Methods:
        plot_all -- plot all lines and markers indicated by data_on, model_on, primary_axis_on, twin_axis_on
        
        plot_primary_data -- plot data along primary y axis
        plot_twin_data -- plot data along self.twin y axis
        plot_primary_model -- plot model curve along primary y axis
        plot_twin_model -- plot model curve along self.twin y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_twin_ghost_data -- plot ghost data along self.twin y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        plot_twin_ghost_model -- plot ghost model curve along self.twin y axis
        
        set_text -- set axis labels and title (not implemented in base class)
        set_axis_colours -- set axis label colours and positions
        
        set_twin_xlim, set_twin_xscale, set_twin_ylim, set_twin_yscale -- set_xlim, set_xscale, set_ylim, and set_yscale for self.twin, with consideration for self.make_twinx
        
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively
        
        d_extend -- provide reasonable lower limit for plot
        u_extend -- provide reasonable upper limit for plot"""
        self.primary = primary
        
        self.data = data
        self.model = model
        self.ghost_data = ghost_data
        self.ghost_data_visibility = ghost_data_visibility
        
        self.make_twinx = make_twinx
        
        self.data_on = data_on
        self.model_on = model_on
        self.primary_axis_on = primary_axis_on
        self.twin_axis_on = twin_axis_on
        
        if self.make_twinx:
            self.twin = self.primary.twinx()
            
        self.primary_data_colour = primary_data_colour
        self.primary_model_colour = primary_model_colour
        self.twin_data_colour = twin_data_colour
        self.twin_model_colour = twin_model_colour
        
        self.ghost_colours = ["#FF00FF", "#FFA000", "#00A0FF"]
        self.ghost_m_colours = ["#FF70FF", "#FFC970", "#70C9FF"]
        
        self.set_axis_colours()
        
    #General plotting
        
    def plot_all(self):
        """Plot all lines and markers indicated by self.data_on, self.model_on, self.primary_axis_on, and self.twin_axis_on. Handles both the main data/model and ghost data/model.
        
        To be called in the plot update method of the PlotFrame."""
        if self.data_on:
            if self.primary_axis_on:
                self.plot_primary_data()
                for i in range(min([len(self.ghost_data_visibility), 3])):
                    self.plot_primary_ghost_data(self.ghost_data[self.ghost_data_visibility[i]], i)
            if self.make_twinx and self.twin_axis_on:
                self.plot_twin_data()
                for i in range(min([len(self.ghost_data_visibility), 3])):
                    self.plot_twin_ghost_data(self.ghost_data[self.ghost_data_visibility[i]], i)
        if self.model_on and self.model != None:
            if self.primary_axis_on:
                self.plot_primary_model()
                for i in range(min([len(self.ghost_data_visibility), 3])):
                    self.plot_primary_ghost_model(self.ghost_data[self.ghost_data_visibility[i]], i)
            if self.make_twinx and self.twin_axis_on:
                self.plot_twin_model()
                for i in range(min([len(self.ghost_data_visibility), 3])):
                    self.plot_twin_ghost_model(self.ghost_data[self.ghost_data_visibility[i]], i)
                    
    #Aesthetics
                    
    def set_text(self):
        """Placeholder function for title, x label, and y label. To be overridden in child classes and called at the end of their __init__ method and in the plot update method in the PlotFrame."""
        raise NotImplementedError
        
    def set_axis_colours(self):
        """Give the axis labels their proper place and colour. To be called in the plot update method in the PlotFrame."""
        self.primary.tick_params(axis = "y", labelcolor = self.primary_data_colour)
        self.primary.yaxis.label.set_color(self.primary_data_colour)
        if self.make_twinx:
            self.twin.tick_params(axis = "y", labelcolor = self.twin_data_colour)
            self.twin.yaxis.tick_right()
            self.twin.yaxis.set_label_position("right")
            self.twin.yaxis.label.set_color(self.twin_data_colour)
            
    #Plotting sub-functions, to be overridden in child classes.
            
    def plot_primary_data(self):
        """Placeholder function for plotting the data along the primary y-axis. Must be overridden in child classes."""
        raise NotImplementedError
        
    def plot_primary_model(self):
        """Placeholder function for plotting the model curve along the primary y-axis. Must be overridden in child classes."""
        raise NotImplementedError

    def plot_twin_data(self):
        """Placeholder function for plotting the data along the self.twin y-axis. Must be overridden in child classes."""
        if self.make_twinx:
            raise NotImplementedError

    def plot_twin_model(self):
        """Placeholder function for plotting the model curve along the self.twin y-axis. Must be overridden in child classes."""
        if self.make_twinx:
            raise NotImplementedError
        
    def plot_primary_ghost_data(self, g, i):
        """Placeholder function for plotting ghost data along the primary y-axis. Must be overridden in child classes.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        raise NotImplementedError
    
    def plot_primary_ghost_model(self, g, i):
        """Placeholder function for plotting ghost model curve along the primary y-axis. Must be overridden in child classes.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        raise NotImplementedError
        
    def plot_twin_ghost_data(self, g, i):
        """Placeholder function for plotting ghost data along the self.twin y-axis. Must be overridden in child classes.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        if self.make_twinx:
            raise NotImplementedError
        
    def plot_twin_ghost_model(self, g, i):
        """Placeholder function for plotting ghost model curve along the self.twin primary y-axis. Must be overridden in child classes.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        if self.make_twinx:
            raise NotImplementedError
        
    #Safe self.twin axis limiting functions
        
    def set_twin_xlim(self, *kwargs):
        """If it exists, set x limits on self.twin. Otherwise, do nothing."""
        if self.make_twinx:
            self.twin.set_xlim(kwargs)
            
    def set_twin_ylim(self, *kwargs):
        """If it exists, set y limits on self.twin. Otherwise, do nothing."""
        if self.make_twinx:
            self.twin.set_ylim(kwargs)
            
    def set_twin_xscale(self, *kwargs):
        """If it exists, set x scaling on self.twin. Otherwise, do nothing."""
        if self.make_twinx:
            self.twin.set_xscale(kwargs)
            
    def set_twin_yscale(self, *kwargs):
        """If it exists, set y scaling on self.twin. Otherwise, do nothing."""
        if self.make_twinx:
            self.twin.set_yscale(kwargs)
            
    #Base limits
    def set_base_limits(self):
        """Set the base limits for all axes. May need to be redefined in child classes. Defaults to reasonable Bode plot limits."""
        self.lim_x = (min(self.data.freq), max(self.data.freq))
        self.lim_y1 = (self.d_extend(min(self.data.amplitude)), self.u_extend(max(self.data.amplitude)))
        self.lim_y2 = (self.d_extend(min(self.data.phase)), self.u_extend(max(self.data.phase)))
            
    #Helper functions
    def d_extend(self, vl):
        """For a given minimum of an axis vl, min(vl), return a logical lower limit for the axis in the plot.
        
        Arguments:
        self
        vl -- list or NumPy array
        
        Returns:
        lower limit for plot (float)"""
        if vl > 0:
            return 0.95*vl
        elif vl == 0:
            return vl
        elif vl < 0:
            return 1.066*vl

    def u_extend(self, vl):
        """For a given maximum of an axis vl, max(vl), return a logical upper limit for the axis in the plot.
         
        Arguments:
        self
        vl -- list or NumPy array
        
        Returns:
        upper limit for plot (float)"""
        if vl > 0:
            return 1.066*vl
        elif vl == 0:
            return vl
        elif vl < 0:
            return 0.95*vl

class ComplexPlaneImpedancePlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = False, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = False, primary_data_colour = "#000000", primary_model_colour = "#DD4444"):
        """Complex plane plot of the impedance (Nyquist plot).
        
        Init arguments: see parent class ImpedancePlot
        
        Methods: see parent class ImpedancePlot
        
        Overridden methods:
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on, primary_data_colour = primary_data_colour, primary_model_colour = primary_model_colour)
        self.set_text()
        self.set_base_limits()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(self.data.real, -self.data.imag, marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(self.model.real, -self.model.imag, marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.data.real, -g.data.imag, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.model.real, -g.model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Complex plane")
        self.primary.set_xlabel("Re[Z] ($\Omega$)")
        self.primary.set_ylabel("-Im[Z] ($\Omega$)")
        
    def set_base_limits(self):
        self.lim_x = (self.d_extend(min(self.data.real)), self.u_extend(max(self.data.real)))
        self.lim_y1 = (self.d_extend(min(-self.data.imag)), self.u_extend(max(-self.data.imag)))
        self.lim_y2 = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        
class ComplexPlaneAdmittancePlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = False, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = False, primary_data_colour = "#000000", primary_model_colour = "#DD4444"):
        """Complex plane plot of the admittance (Nyquist plot).
        
        Init arguments: see parent class ImpedancePlot
        
        Methods: see parent class ImpedancePlot
        
        Overridden methods:
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on, primary_data_colour = primary_data_colour, primary_model_colour = primary_model_colour)
        self.set_text()
        self.set_base_limits()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(1/self.data.real, -1/self.data.imag, marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(1/self.model.real, -1/self.model.imag, marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(1/g.data.real, -1/g.data.imag, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(1/g.model.real, -1/g.model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Complex plane")
        self.primary.set_xlabel("Re[Y] ($\Omega^{-1}$)")
        self.primary.set_ylabel("-Im[Y] ($\Omega^{-1}$)")
        
    def set_base_limits(self):
        self.lim_x = (self.d_extend(min(1/self.data.real)), self.u_extend(max(1/self.data.real)))
        self.lim_y1 = (self.d_extend(min(-1/self.data.imag)), self.u_extend(max(-1/self.data.imag)))
        self.lim_y2 = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        
class BodePhaseAmplitudePlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = True, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = True):
        """Bode amplitude/phase plot.
        
        Init arguments: see parent class ImpedancePlot
        
        Methods: see parent class ImpedancePlot
        
        Overridden methods:
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        plot_twin_data -- plot data along twin y axis
        plot_twin_model -- plot model curve along twin y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        plot_twin_ghost_data -- plot ghost data along twin y axis
        plot_twin_ghost_model -- plot ghost model curve along twin y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on)
        self.set_text()
        self.set_base_limits()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(self.data.freq, self.data.amplitude, marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(self.model.freq, self.model.amplitude, marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.data.freq, g.data.amplitude, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.model.freq, g.model.amplitude, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def plot_twin_data(self):
        """Function for plotting the data along the self.twin y-axis."""
        self.twin.plot(self.data.freq, self.data.phase*180/np.pi, marker = ".", linestyle = "None", color = self.twin_data_colour)
        
    def plot_twin_model(self):
        """Function for plotting the model curve along the self.twin y-axis."""
        self.twin.plot(self.model.freq, self.model.phase*180/np.pi, marker = "None", linestyle = "-", color = self.twin_model_colour)

    def plot_twin_ghost_data(self, g, i):
        """Function for plotting ghost data along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.data.freq, g.data.phase*180/np.pi, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_twin_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.model.freq, g.model.phase*180/np.pi, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Bode amplitude and phase")
        self.primary.set_xlabel("Frequency (Hz)")
        self.primary.set_ylabel("|Z| ($\Omega$)")
        self.twin.set_ylabel("$\phi$ (°)")
        
    def set_base_limits(self):
        self.lim_x = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        self.lim_y1 = (self.d_extend(min(self.data.amplitude)), self.u_extend(max(self.data.amplitude)))
        self.lim_y2 = (self.d_extend(min(self.data.phase*180/np.pi)), self.u_extend(max(self.data.phase*180/np.pi)))
        
class AdmittanceFrequencyPlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = True, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = True):
        """Y', Y'' vs. frequency plot.
        
        Init arguments: see parent class ImpedancePlot
        
        Methods: see parent class ImpedancePlot
        
        Overridden methods:
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        plot_twin_data -- plot data along twin y axis
        plot_twin_model -- plot model curve along twin y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        plot_twin_ghost_data -- plot ghost data along twin y axis
        plot_twin_ghost_model -- plot ghost model curve along twin y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on)
        self.set_text()
        self.set_base_limits()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(self.data.freq, 1/self.data.real, marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(self.model.freq, 1/self.model.real, marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.data.freq, 1/g.data.real, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.model.freq, 1/g.model.real, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def plot_twin_data(self):
        """Function for plotting the data along the self.twin y-axis."""
        self.twin.plot(self.data.freq, -1/self.data.imag, marker = ".", linestyle = "None", color = self.twin_data_colour)
        
    def plot_twin_model(self):
        """Function for plotting the model curve along the self.twin y-axis."""
        self.twin.plot(self.model.freq, -1/self.model.imag, marker = "None", linestyle = "-", color = self.twin_model_colour)

    def plot_twin_ghost_data(self, g, i):
        """Function for plotting ghost data along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.data.freq, -1/g.data.imag, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_twin_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.model.freq, -1/g.model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Admittance")
        self.primary.set_xlabel("Frequency (Hz)")
        self.primary.set_ylabel("Y\' ($\Omega^{-1}$)")
        self.twin.set_ylabel("Y\'\' ($\Omega^{-1}$)")
        
    def set_base_limits(self):
        self.lim_x = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        self.lim_y1 = (self.d_extend(min(1/self.data.real)), self.u_extend(max(1/self.data.real)))
        self.lim_y2 = (self.d_extend(min(-1/self.data.imag)), self.u_extend(max(-1/self.data.imag)))
        
class ImpedanceFrequencyPlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = True, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = True):
        """Z', Z'' vs. frequency plot.
        
        Init arguments: see parent class ImpedancePlot
        
        Methods: see parent class ImpedancePlot
        
        Overridden methods:
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        plot_twin_data -- plot data along twin y axis
        plot_twin_model -- plot model curve along twin y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        plot_twin_ghost_data -- plot ghost data along twin y axis
        plot_twin_ghost_model -- plot ghost model curve along twin y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on)
        self.set_text()
        self.set_base_limits()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(self.data.freq, self.data.real, marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(self.model.freq, self.model.real, marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.data.freq, g.data.real, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.model.freq, g.model.real, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def plot_twin_data(self):
        """Function for plotting the data along the self.twin y-axis."""
        self.twin.plot(self.data.freq, -self.data.imag, marker = ".", linestyle = "None", color = self.twin_data_colour)
        
    def plot_twin_model(self):
        """Function for plotting the model curve along the self.twin y-axis."""
        self.twin.plot(self.model.freq, -self.model.imag, marker = "None", linestyle = "-", color = self.twin_model_colour)

    def plot_twin_ghost_data(self, g, i):
        """Function for plotting ghost data along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.data.freq, -g.data.imag, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_twin_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.model.freq, -g.model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Impedance")
        self.primary.set_xlabel("Frequency (Hz)")
        self.primary.set_ylabel("Z\' ($\Omega$)")
        self.twin.set_ylabel("Z\'\' ($\Omega$)")
        
    def set_base_limits(self):
        self.lim_x = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        self.lim_y1 = (self.d_extend(min(self.data.real)), self.u_extend(max(self.data.real)))
        self.lim_y2 = (self.d_extend(min(-self.data.imag)), self.u_extend(max(-self.data.imag)))
        
class ConductivityFrequencyPlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = True, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = True, sample_area = 1, sample_thickness = 1):
        """sigma', sigma'' vs. frequency plot.
        
        Init arguments: see parent class ImpedancePlot, and:
        sample_area -- area of sample/electrode interface in cm^2
        sample_thickness -- thickness of sample in mm
        
        Other attributes:
        geom -- sample_thickness/sample_area in units of 1/cm
        
        Methods: see parent class ImpedancePlot
        
        Overridden methods:
        plot_all -- plot all data
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        plot_twin_data -- plot data along twin y axis
        plot_twin_model -- plot model curve along twin y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        plot_twin_ghost_data -- plot ghost data along twin y axis
        plot_twin_ghost_model -- plot ghost model curve along twin y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on)
        self.sample_area = sample_area
        self.sample_thickness = sample_thickness
        self.geom = 0.1*sample_thickness/sample_area
        self.set_text()
        self.set_base_limits()
        
    def plot_all(self):
        """Plot all data and update the sample geometry."""
        self.geom = 0.1*self.sample_thickness/(self.sample_area)
        super().plot_all()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(self.data.freq, self.geom/self.data.real, marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(self.model.freq, self.geom/self.model.real, marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.data.freq, self.geom/g.data.real, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.model.freq, self.geom/g.model.real, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def plot_twin_data(self):
        """Function for plotting the data along the self.twin y-axis."""
        self.twin.plot(self.data.freq, -self.geom/self.data.imag, marker = ".", linestyle = "None", color = self.twin_data_colour)
        
    def plot_twin_model(self):
        """Function for plotting the model curve along the self.twin y-axis."""
        self.twin.plot(self.model.freq, -self.geom/self.model.imag, marker = "None", linestyle = "-", color = self.twin_model_colour)

    def plot_twin_ghost_data(self, g, i):
        """Function for plotting ghost data along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.data.freq, -self.geom/g.data.imag, marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_twin_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.model.freq, -self.geom/g.model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Conductivity")
        self.primary.set_xlabel("Frequency (Hz)")
        self.primary.set_ylabel("$\sigma$\' (S cm$^{-1}$)")
        self.twin.set_ylabel("$\sigma$\'\' (S cm$^{-1}$)")
        
    def set_base_limits(self):
        self.geom = 0.1*self.sample_thickness/(self.sample_area)
        self.lim_x = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        self.lim_y1 = (self.d_extend(min(self.geom/self.data.real)), self.u_extend(max(self.geom/self.data.real)))
        self.lim_y2 = (self.d_extend(min(-self.geom/self.data.imag)), self.u_extend(max(-self.geom/self.data.imag)))

class PerimttivityFrequencyPlot(ImpedancePlot):
    def __init__(self, primary, data, model, ghost_data, ghost_data_visibility, make_twinx = True, data_on = True, model_on = True, primary_axis_on = True, twin_axis_on = True, sample_area = 1, sample_thickness = 1):
        """sigma', sigma'' vs. frequency plot.
        
        Init arguments: see parent class ImpedancePlot, and:
        sample_area -- area of sample/electrode interface in cm^2
        sample_thickness -- thickness of sample in mm
        
        Other attributes:
        geom -- sample_thickness/sample_area in units of 1/m
        e0 -- electric constant, 8.854e-12 F/m
        permittivity_data -- complex NumPy array of permittivities
        permittivity_model
        
        Methods: see parent class ImpedancePlot, and:
        permittivity -- calculate self.permittivity_data and self.permittivity_model
        permittivity_generic -- calculate the complex permittivity of a dataSet
        
        Overridden methods:
        plot_all -- plot all data
        
        plot_primary_data -- plot data along primary y axis
        plot_primary_model -- plot model curve along primary y axis
        plot_twin_data -- plot data along twin y axis
        plot_twin_model -- plot model curve along twin y axis
        
        plot_primary_ghost_data -- plot ghost data along primary y axis
        plot_primary_ghost_model -- plot ghost model curve along primary y axis
        plot_twin_ghost_data -- plot ghost data along twin y axis
        plot_twin_ghost_model -- plot ghost model curve along twin y axis
        
        set_text -- set axis labels and title
        set_base_limits -- set self.lim_x, self.lim_y1, and self.lim_y2; these are (low, high) Tuples of plot limits that provide a view of the entire dataset for x, primary y, and twin y, respectively"""
        super().__init__(primary, data, model, ghost_data, ghost_data_visibility, make_twinx = make_twinx, data_on = data_on, model_on = model_on, primary_axis_on = primary_axis_on, twin_axis_on = twin_axis_on)
        self.geom = 0.001*sample_thickness/0.01*sample_area
        self.sample_area = sample_area
        self.sample_thickness = sample_thickness
        self.e0 = 8.854e-12
        self.set_text()
        self.permittivity()
        self.set_base_limits()
        
    def permittivity(self):
        """Calculate the complex permittivity and save the data values to self.permittivity_data and the model values to self.permittivity_model."""
        sigma_data = self.geom/(self.data.real + 1j*self.data.imag)
        self.permittivity_data = sigma_data/(1j*2*np.pi*self.data.freq*self.e0)
        if self.model != None:
            sigma_model = self.geom/(self.model.real + 1j*self.model.imag)
            self.permittivity_model = sigma_model/(1j*2*np.pi*self.model.freq*self.e0)
        
    def permittivity_generic(self, ds):
        """Calculate complex permittivity data for a dataSet ds.
        
        Arguments:
        self
        ds -- dataSet
        
        Returns:
        Complex permittivity (NumPy array)"""
        sigma = self.geom/(ds.real + 1j*ds.imag)
        return sigma/(1j*ds.freq*2*np.pi*self.e0)
        
    def plot_all(self):
        """Plot all data, but first calculate the new permittivity."""
        self.geom = 0.001*self.sample_thickness/0.01*self.sample_area
        self.permittivity()
        super().plot_all()
    
    def plot_primary_data(self):
        """Function for plotting the data along the primary y-axis."""
        self.primary.plot(self.data.freq, np.real(self.permittivity_data), marker = ".", linestyle = "None", color = self.primary_data_colour)
        
    def plot_primary_model(self):
        """Function for plotting the model curve along the primary y-axis."""
        self.primary.plot(self.model.freq, np.real(self.permittivity_model), marker = "None", linestyle = "-", color = self.primary_model_colour)

    def plot_primary_ghost_data(self, g, i):
        """Function for plotting ghost data along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.data.freq, np.real(self.permittivity_generic(g.data)), marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_primary_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the primary y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.primary.plot(g.model.freq, np.real(self.permittivity_generic(g.model)), marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def plot_twin_data(self):
        """Function for plotting the data along the self.twin y-axis."""
        self.twin.plot(self.data.freq, -np.imag(self.permittivity_data), marker = ".", linestyle = "None", color = self.twin_data_colour)
        
    def plot_twin_model(self):
        """Function for plotting the model curve along the self.twin y-axis."""
        self.twin.plot(self.model.freq, -np.imag(self.permittivity_model), marker = "None", linestyle = "-", color = self.twin_model_colour)

    def plot_twin_ghost_data(self, g, i):
        """Function for plotting ghost data along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.data.freq, -np.imag(self.permittivity_generic(g.data)), marker = ".", linestyle = "None", color = self.ghost_colours[i])
    
    def plot_twin_ghost_model(self, g, i):
        """Function for plotting ghost model curve along the self.twin y-axis.
        
        Arguments:
        self
        g -- ghost data expandedDataSet
        i -- index to use for colour"""
        self.twin.plot(g.model.freq, -np.imag(self.permittivity_generic(g.model)), marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
        
    def set_text(self):
        self.primary.set_title("Permittivity")
        self.primary.set_xlabel("Frequency (Hz)")
        self.primary.set_ylabel("$\epsilon$\'")
        self.twin.set_ylabel("$\epsilon$\'\'")
        
    def set_base_limits(self):
        self.geom = 0.001*self.sample_thickness/0.01*self.sample_area
        self.permittivity()
        self.lim_x = (self.d_extend(min(self.data.freq)), self.u_extend(max(self.data.freq)))
        self.lim_y1 = (self.d_extend(min(np.real(self.permittivity_data))), self.u_extend(max(np.real(self.permittivity_data))))
        self.lim_y2 = (self.d_extend(min(-np.imag(self.permittivity_data))), self.u_extend(max(-np.imag(self.permittivity_data))))

###################
##PLOTTING CANVAS##
###################

class PlotFrame(ttk.Frame):
    def __init__(self, mdim, data, ghost_data, ghost_data_visibility):
        """Controls the internal layout of the Matplotlib plots.
        
        Init arguments:
        mdim -- master dimensions; tuple or list of (width, height)
        data -- dataSet containing measured data
        ghost_data -- dictionary of {'dataset name': ecm_history.expandedDataSet}; holds all data and models saved to History
        ghost_data_visibility -- list of names in ghost_data; all names in this list are of dataSets that should be visible
        
        Attributes:
        fig -- Matplotlib Figure
        lhs, rhs -- axes on fig
        left_plot, right_plot -- ImpedancePlots belonging to self.lhs and self.rhs
        title -- plot title
        canvas -- plotting canvas
        toolbar -- navigation toolbar
        limiter -- limiter for when the toolbar is being used by the user
        
        logRY1 -- Boolean, indicates if primary RHS axis should be log-scaled
        logRY2 -- Boolean, indicates if secondary RHS axis should be log-scaled
        datavis -- Boolean, data visibility
        fitvis -- Boolean, model visibility
        visRY1 -- Boolean, primary RHS axis visibility
        visRY2 -- Boolean, secondary RHS axis visibility
        
        plot_types -- dict containing the different types of plots
        needs_geom -- dict with the same keys as self.plot_types, but keys are Booleans that indicate if the plot requires geometric information
        rhs_type -- type of plot on the right (key in plot_types)
        prev_rhs_type -- previous type of plot on the right (key in plot_types)
        lhs_type -- type of plot on the left (key in plot_types)
        prev_lhs_type -- previous type of plot on the left (key in plot_types)
                
        ghost_colours -- list of color strings for History data
        ghost_m_colours -- list of color strings for History models
        mfreq_freq -- NumPy array of frequencies of points marked by the 'mark frequencies that are integer powers of 10' option
        mfreq_real -- NumPy array of real components of marked points
        mfreq_imag -- NumPy array of imaginary components of marked points
        decades -- Boolean, whether to indicate frequencies with decades (True) or frequency values (False); default False
        
        sample_thickness -- thickness of sample in mm
        sample_area -- area of sample in cm^2
        
        Methods:
        makeFrame -- build up the PlotFrame
        d_extend, u_extend -- used to determine logical lower and upper limits, respectively, for a given plot
        applyFrequencyMarking -- mark frequencies that are integer powers of 10 in the complex plane plot
        updatePlots -- clear and redraw all plots"""
        super().__init__() #Initialise self as ttk.Frame.
        self.makeFrame(mdim, data, ghost_data, ghost_data_visibility) #Expansion code, turning itself into a PlotFrame.

    def makeFrame(self, master_dim, data, ghost_data, ghost_data_visibility):
        """Create the figure, figure canvas and plot toolbar. Then update the plots.
        
        Arguments:
        self
        master_dim -- master dimensions; tuple or list of (width, height)
        data -- dataSet containing measured data
        ghost_data -- dictionary of {'dataset name': ecm_history.expandedDataSet}; holds all data and models saved to History
        ghost_data_visibility -- list of names in ghost_data; all names in this list are of dataSets that should be visible"""
        self.fig = fg.Figure(figsize = (17*master_dim[0]/1280, 5*master_dim[1]/540), dpi = 82) #Create a Matplotlib figure based on the master's dimensions, which depends on the display size.
        self.lhs = self.fig.add_subplot(1, 2, 1) #Create the LHS subplot (commonly called Nyquist plot).
        self.rhs = self.fig.add_subplot(1, 2, 2) #Create the RHS subplot
        self.title = "" #Title.
        self.canvas = btk.FigureCanvasTkAgg(self.fig, self) #Create canvas on which to draw the Matplotlib figure, self.fig.
        self.toolbar = btk.NavigationToolbar2Tk(self.canvas, self) #Add navigation buttons to the canvas.
        self.toolbar.update() #Necessary.
        self.canvas.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True) #Place the canvas on the frame.
        self.limiter = limiter(enabled = False) #Create a limiter to set plot boundaries.
        self.logRY1 = True #Setting for logarithmic scaling of the amplitude. Or in the AB plot, sigma'.
        self.logRY2 = False #Setting for logarithmic scaling of the phase. Really meant for the AB plot, where it scales sigma''.
        self.datavis = True #Data visibility
        self.fitvis = True #Model curve visibility
        self.visRY1 = True #Amplitude/real AB visibility
        self.visRY2 = True #Phase/imaginary AB visibility
        self.lhs_type = "Complex plane Z"
        self.rhs_type = "Bode amplitude/phase"
        self.prev_lhs_type = "Complex plane Z"
        self.prev_rhs_type = "Bode amplitude/phase"
        self.ghost_colours = ["#FF00FF", "#FFA000", "#00A0FF"]
        self.ghost_m_colours = ["#FF70FF", "#FFC970", "#70C9FF"]
        self.mfreq_freq = np.array([])
        self.mfreq_real = np.array([])
        self.mfreq_imag = np.array([])
        self.decades = False
        self.sample_thickness = 1
        self.sample_area = 1
        self.plot_types = {"Complex plane Z": ComplexPlaneImpedancePlot, "Complex plane Y": ComplexPlaneAdmittancePlot, "Bode amplitude/phase": BodePhaseAmplitudePlot, "YY vs. f": AdmittanceFrequencyPlot, "ZZ vs. f": ImpedanceFrequencyPlot, "sigma vs. f": ConductivityFrequencyPlot, "epsilon vs. f": PerimttivityFrequencyPlot}
        self.needs_geom = {"Complex plane Z": False, "Complex plane Y": False, "Bode amplitude/phase": False, "YY vs. f": False, "ZZ vs. f": False, "sigma vs. f": True, "epsilon vs. f": True}
        if self.needs_geom[self.lhs_type]:
            self.left_plot = self.plot_types[self.lhs_type](self.lhs, data, None, ghost_data, ghost_data_visibility, sample_area = self.sample_area, sample_thickness = self.sample_thickness)
        else:
            self.left_plot = self.plot_types[self.lhs_type](self.lhs, data, None, ghost_data, ghost_data_visibility)
        if self.needs_geom[self.rhs_type]:
            self.right_plot = self.plot_types[self.rhs_type](self.rhs, data, None, ghost_data, ghost_data_visibility, sample_area = self.sample_area, sample_thickness = self.sample_thickness)
        else:
            self.right_plot = self.plot_types[self.rhs_type](self.rhs, data, None, ghost_data, ghost_data_visibility)
        self.fig.subplots_adjust(wspace = 0.3)
        self.updatePlots(data, ghost_data, ghost_data_visibility) #Update all plots with the given data.

    #Helper functions
    def d_extend(self, vl):
        """For a given minimum of an axis vl, min(vl), return a logical lower limit for the axis in the plot.
        
        Arguments:
        self
        vl -- list or NumPy array
        
        Returns:
        lower limit for plot (float)"""
        if vl > 0:
            return 0.95*vl
        elif vl == 0:
            return vl
        elif vl < 0:
            return 1.066*vl

    def u_extend(self, vl):
        """For a given maximum of an axis vl, max(vl), return a logical upper limit for the axis in the plot.
         
        Arguments:
        self
        vl -- list or NumPy array
        
        Returns:
        upper limit for plot (float)"""
        if vl > 0:
            return 1.066*vl
        elif vl == 0:
            return vl
        elif vl < 0:
            return 0.95*vl

    def applyFrequencyMarking(self):
        """Based on which frequencies were found to be present in the measured dataSet (see DECiM core), draw circles around datapoints whose frequencies are integer powers of 10, write their frequencies in the plot and draw lines from the text to the circles."""
        if self.lhs_type == "Complex plane Z":
            self.lhs.plot(self.mfreq_real, -self.mfreq_imag, marker = "o", fillstyle = "none", linestyle = "None", color = "#000000", markersize = 12)
        elif self.lhs_type == "Complex plane Y":
            self.lhs.plot(1/self.mfreq_real, -1/self.mfreq_imag, marker = "o", fillstyle = "none", linestyle = "None", color = "#000000", markersize = 12)
        if self.decades:
            mfreq_labels = {}
            for i in range(-6, 13, 1):
                mfreq_labels[10**i] = str(i)
        else:
            mfreq_labels = {0.000001: "1 $\mu$Hz", 0.00001: "10 $\mu$Hz", 0.0001: "0.1 mHz", 0.001: "1 mHz", 0.01: "10 mHz", 0.1: "0.1 Hz", 1: "1 Hz", 10: "10 Hz", 100: "100 Hz", 1000: "1 kHz", 10000: "10 kHz", 100000: "100 kHz", 1000000: "1 MHz", 10000000: "10 MHz", 100000000: "100 MHz", 1000000000: "1 GHz", 10000000000: "10 GHz", 100000000000: "100 GHz", 1000000000000: "1 THz"}
        if self.limiter.enabled:
            where_x = self.limiter.real
            where_y = self.limiter.imag
        else:
            where_x = self.lhs.get_xlim()
            where_y = self.lhs.get_ylim()
        dx, dy = where_x[1] - where_x[0], where_y[1] - where_y[0]
        for m in range(len(self.mfreq_freq)):
            if self.lhs_type == "Complex plane Z":
                if self.decades:
                    self.lhs.text(self.mfreq_real[m] + 0.05*dx, -self.mfreq_imag[m] - 0.075*dy, mfreq_labels[self.mfreq_freq[m]])
                    self.lhs.plot([self.mfreq_real[m] + 0.02*dx, self.mfreq_real[m] + 0.045*dx], [-self.mfreq_imag[m] - 0.015*dy, -self.mfreq_imag[m] - 0.025*dy], marker = "None", linestyle = "-", linewidth = 1, color = "#000000")
                else:
                    self.lhs.text(self.mfreq_real[m] + 0.05*dx, -self.mfreq_imag[m] - 0.05*dy, mfreq_labels[self.mfreq_freq[m]])
                    self.lhs.plot([self.mfreq_real[m] + 0.02*dx, self.mfreq_real[m] + 0.045*dx], [-self.mfreq_imag[m] - 0.015*dy, -self.mfreq_imag[m] - 0.04*dy], marker = "None", linestyle = "-", linewidth = 1, color = "#000000")
            elif self.lhs_type == "Complex plane Y":
                if self.decades:
                    self.lhs.text(1/self.mfreq_real[m] + 0.05*dx, -1/self.mfreq_imag[m] - 0.075*dy, mfreq_labels[self.mfreq_freq[m]])
                    self.lhs.plot([1/self.mfreq_real[m] + 0.02*dx, 1/self.mfreq_real[m] + 0.045*dx], [-1/self.mfreq_imag[m] - 0.015*dy, -1/self.mfreq_imag[m] - 0.025*dy], marker = "None", linestyle = "-", linewidth = 1, color = "#000000")
                else:
                    self.lhs.text(1/self.mfreq_real[m] + 0.05*dx, -1/self.mfreq_imag[m] - 0.05*dy, mfreq_labels[self.mfreq_freq[m]])
                    self.lhs.plot([1/self.mfreq_real[m] + 0.02*dx, 1/self.mfreq_real[m] + 0.045*dx], [-1/self.mfreq_imag[m] - 0.015*dy, -1/self.mfreq_imag[m] - 0.04*dy], marker = "None", linestyle = "-", linewidth = 1, color = "#000000")

    def updatePlots(self, data, ghost_data, ghost_data_visibility, model = None):
        """Clear and redraw the plots. All the checks for which data should be visible are handled here. This is also where the model curve finally comes in.
        
        Arguments:
        self
        master_dim -- master dimensions; tuple or list of (width, height)
        data -- dataSet containing measured data
        ghost_data -- dictionary of {'dataset name': ecm_history.expandedDataSet}; holds all data and models saved to History
        ghost_data_visibility -- list of names in ghost_data; all names in this list are of dataSets that should be visible
        model -- None or a dataSet containing the model curve"""
        #Reset plot types
        if self.rhs_type != self.prev_rhs_type or self.lhs_type != self.prev_lhs_type:
            self.fig.clf()
            self.lhs = self.fig.add_subplot(1, 2, 1) #Create the LHS subplot (commonly called Nyquist plot).
            self.rhs = self.fig.add_subplot(1, 2, 2) #Create the RHS subplot
            del self.right_plot
            if self.needs_geom[self.rhs_type]:
                self.right_plot = self.plot_types[self.rhs_type](self.rhs, data, None, ghost_data, ghost_data_visibility, sample_area = self.sample_area, sample_thickness = self.sample_thickness)
            else:
                self.right_plot = self.plot_types[self.rhs_type](self.rhs, data, None, ghost_data, ghost_data_visibility)
            del self.left_plot
            if self.needs_geom[self.lhs_type]:
                self.left_plot = self.plot_types[self.lhs_type](self.lhs, data, None, ghost_data, ghost_data_visibility, sample_area = self.sample_area, sample_thickness = self.sample_thickness)
            else:
                self.left_plot = self.plot_types[self.lhs_type](self.lhs, data, None, ghost_data, ghost_data_visibility)
        
        #Clear all subplots
        self.left_plot.primary.cla()
        self.right_plot.primary.cla()
        self.right_plot.twin.cla()

        #Set some titles and labels
        self.fig.suptitle(self.title)

        #Log-scaling of the RHS plot
        self.right_plot.primary.set_xscale("log")
        if self.logRY1:
            self.right_plot.primary.set_yscale("log")
        if self.logRY2:
            self.right_plot.twin.set_yscale("log")
        
        #RHS visibility
        self.right_plot.primary_axis_on = self.visRY1
        self.right_plot.twin_axis_on = self.visRY2
        
        #Update plots
        for panel in [self.left_plot, self.right_plot]:
            #Visibilities
            panel.data_on = self.datavis
            panel.model_on = self.fitvis
            #Data & model update
            panel.data = data
            panel.model = model
            #Text update
            panel.set_text()
            panel.set_axis_colours()
            #Limits update
            panel.set_base_limits()
            #Plotted data/model/ghost datasets/ghost models
            panel.plot_all()

        #Apply frequency marking in the LHS plot, if enabled
        if self.datavis:
            #Frequency labelling
            if len(self.mfreq_freq) > 0:
                self.applyFrequencyMarking()

        #Set the subplots' boundaries.
        if self.limiter.enabled: #If the limiter is enabled, use it to limit the subplots' boundaries.
            self.left_plot.primary.set(xlim = self.limiter.real)
            self.left_plot.primary.set(ylim = self.limiter.imag)
            self.right_plot.primary.set(xlim = self.limiter.freq)
            self.right_plot.primary.set(ylim = self.limiter.amp)
            self.right_plot.twin.set(ylim = self.limiter.phase)
        else: #If the limiter is not enabled, apply automatic plot boundaries (defined a few lines above) via base_lims.
            self.left_plot.primary.set(xlim = self.left_plot.lim_x)
            self.left_plot.primary.set(ylim = self.left_plot.lim_y1)
            self.right_plot.primary.set(xlim = self.right_plot.lim_x)
            self.right_plot.primary.set(ylim = self.right_plot.lim_y1)
            self.right_plot.twin.set(ylim = self.right_plot.lim_y2)

        #Draw the new model, along with the data, on the canvas.
        self.canvas.draw()

##########################
##SAMPLE GEOMETRY WINDOW##
##########################

class GeometryWindow(tk.Toplevel):
    def __init__(self, plotframe):
        """Sample geometry window. Used to set sample thickness and area
        
        Init arguments becoming attributes under the same name:
        plotframe -- PlotFrame on which to alter plots
        
        Other attributes:
        area -- tk.StringVar holding the area in cm^2
        thickness - tk.StringVar holding the thickness in mm
        
        Methods:
        make_UI -- create the UI
        terminate -- close the window and transfer the sample thickness"""
        super().__init__()
        self.title("Sample geometry")
        self.width = int(self.winfo_screenwidth()*0.25)
        self.height = int(self.winfo_screenheight()*0.15)
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        
        self.plotframe = plotframe
        
        self.make_UI()
        
    def make_UI(self):
        """Create the GeometryWindow UI."""
        self.thickness = tk.StringVar()
        self.thickness_frame = ttk.Frame(self)
        self.thickness_frame.pack(side = tk.TOP, anchor = tk.CENTER)
        
        self.area = tk.StringVar()
        self.area_frame = ttk.Frame(self)
        self.area_frame.pack(side = tk.TOP, anchor = tk.CENTER)
        
        self.conclude_frame = ttk.Frame(self)
        self.conclude_frame.pack(side = tk.TOP, anchor = tk.CENTER)
        
        self.thickness_label = tk.Label(self.thickness_frame, text = "Sample thickness (mm): ")
        self.thickness_label.pack(side = tk.LEFT, anchor = tk.CENTER)
        
        self.area_label = tk.Label(self.area_frame, text = "Sample area (cm^2): ")
        self.area_label.pack(side = tk.LEFT, anchor = tk.CENTER)
        
        self.thickness_entry = tk.Entry(self.thickness_frame, textvariable = self.thickness)
        self.thickness_entry.pack(side = tk.LEFT, anchor = tk.CENTER)
        
        self.area_entry = tk.Entry(self.area_frame, textvariable = self.area)
        self.area_entry.pack(side = tk.LEFT, anchor = tk.CENTER)
        
        self.end_button = tk.Button(self.conclude_frame, text = "Set sample dimensions and close", command = self.terminate)
        self.end_button.pack(side = tk.TOP, anchor = tk.CENTER)
        
        self.thickness.set("1")
        self.area.set("1")
        
    def terminate(self):
        """Set self.plotframe's sample thickness and area and mark the current geometry-dependent plots for resetting."""
        self.plotframe.sample_thickness = float(self.thickness.get())
        self.plotframe.sample_area = float(self.area.get())
        if self.plotframe.needs_geom[self.plotframe.rhs_type]:
            self.plotframe.prev_rhs_type = "None"
        if self.plotframe.needs_geom[self.plotframe.lhs_type]:
            self.plotframe.prev_lhs_type = "None"
        self.destroy()