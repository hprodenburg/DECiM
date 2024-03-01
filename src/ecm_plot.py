"""Part of DECiM. This file contains the plotting canvas class and some additional plotting-related code. Last modified 1 March 2024 by Henrik Rodenburg.

Classes:
limiter -- handles the upper and lower limits of the plots
PlotFrame -- ttk.Frame with the plot canvas and toolbar"""

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
        complex_plane, self.ampltiude, self.phase -- the different axes on fig
        title -- plot title
        canvas -- plotting canvas
        toolbar -- navigation toolbar
        limiter -- limiter for when the toolbar is being used by the user
        base_lims -- limiter for when the toolbar is not being used by the user; contains limits that cover the extent of the whole measured dataSet
        
        logamp -- Boolean, indicates if amplitude or real admittance axis should be log-scaled
        logphase -- Boolean, indicates if phase or imaginary admittance axis should be log-scaled
        datavis -- Boolean, data visibility
        fitvis -- Boolean, model visibility
        ampvis -- Boolean, amplitude / real admittance visibility
        phavis -- Boolean, phase / imaginary admittance visibility
        admittance_plot -- Boolean, indicates if the Bode or real/imaginary admittance plot should be shown
        
        ghost_colours -- list of color strings for History data
        ghost_m_colours -- list of color strings for History models
        mfreq_freq -- NumPy array of frequencies of points marked by the 'mark frequencies that are integer powers of 10' option
        mfreq_real -- NumPy array of real components of marked points
        mfreq_imag -- NumPy array of imaginary components of marked points
        
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
        self.complex_plane = self.fig.add_subplot(1, 2, 1) #Create the complex plane subplot (commonly called Nyquist plot).
        self.amplitude = self.fig.add_subplot(1, 2, 2) #Create the Bode amplitude subplot.
        self.phase = self.amplitude.twinx() #Create the Bode phase plot in the same subplot at the Bode amplitude; they share a horizontal axis.
        self.title = "" #Title.
        self.canvas = btk.FigureCanvasTkAgg(self.fig, self) #Create canvas on which to draw the Matplotlib figure, self.fig.
        self.toolbar = btk.NavigationToolbar2Tk(self.canvas, self) #Add navigation buttons to the canvas.
        self.toolbar.update() #Necessary.
        self.canvas.get_tk_widget().pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True) #Place the canvas on the frame.
        self.limiter = limiter(enabled = False) #Create a limiter to set plot boundaries.
        self.base_lims = limiter(enabled = True)
        self.logamp = True #Setting for logarithmic scaling of the amplitude. Or in the AB plot, sigma'.
        self.logphase = False #Setting for logarithmic scaling of the phase. Really meant for the AB plot, where it scales sigma''.
        self.datavis = True #Data visibility
        self.fitvis = True #Model curve visibility
        self.ampvis = True #Amplitude/real AB visibility
        self.phavis = True #Phase/imaginary AB visibility
        self.admittance_plot = False #Plot the real and imaginary components of the admittance against frequency instead of the amplitude and phase of the impedance.
        self.ghost_colours = ["#FF00FF", "#FFA000", "#00A0FF"]
        self.ghost_m_colours = ["#FF70FF", "#FFC970", "#70C9FF"]
        self.mfreq_freq = np.array([])
        self.mfreq_real = np.array([])
        self.mfreq_imag = np.array([])
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
        self.complex_plane.plot(self.mfreq_real, -self.mfreq_imag, marker = "o", fillstyle = "none", linestyle = "None", color = "#000000", markersize = 12)
        mfreq_labels = {0.000001: "1 $\mu$Hz", 0.00001: "10 $\mu$Hz", 0.0001: "0.1 mHz", 0.001: "1 mHz", 0.01: "10 mHz", 0.1: "0.1 Hz", 1: "1 Hz", 10: "10 Hz", 100: "100 Hz", 1000: "1 kHz", 10000: "10 kHz", 100000: "100 kHz", 1000000: "1 MHz", 10000000: "10 MHz", 100000000: "100 MHz", 1000000000: "1 GHz", 10000000000: "10 GHz", 100000000000: "100 GHz", 1000000000000: "1 THz"}
        if self.limiter.enabled:
            where_x = self.limiter.real
            where_y = self.limiter.imag
        else:
            where_x = self.complex_plane.get_xlim()
            where_y = self.complex_plane.get_ylim()
        dx, dy = where_x[1] - where_x[0], where_y[1] - where_y[0]
        for m in range(len(self.mfreq_freq)):
            self.complex_plane.text(self.mfreq_real[m] + 0.05*dx, -self.mfreq_imag[m] - 0.05*dy, mfreq_labels[self.mfreq_freq[m]])
            self.complex_plane.plot([self.mfreq_real[m] + 0.02*dx, self.mfreq_real[m] + 0.045*dx], [-self.mfreq_imag[m] - 0.015*dy, -self.mfreq_imag[m] - 0.04*dy], marker = "None", linestyle = "-", linewidth = 1, color = "#000000")

    def updatePlots(self, data, ghost_data, ghost_data_visibility, model = None):
        """Clear and redraw the plots. All the checks for which data should be visible are handled here. This is also where the model curve finally comes in.
        
        Arguments:
        self
        master_dim -- master dimensions; tuple or list of (width, height)
        data -- dataSet containing measured data
        ghost_data -- dictionary of {'dataset name': ecm_history.expandedDataSet}; holds all data and models saved to History
        ghost_data_visibility -- list of names in ghost_data; all names in this list are of dataSets that should be visible
        model -- None or a dataSet containing the model curve"""
        #Clear all subplots
        self.complex_plane.cla()
        self.amplitude.cla()
        self.phase.cla()

        #Set some titles and labels
        self.fig.suptitle(self.title)
        self.complex_plane.set_title("Complex plane")
        self.complex_plane.set_xlabel("Re[Z] (Ohm)")
        self.amplitude.set_xlabel("Frequency (Hz)")
        self.phase.set_xlabel("Frequency (Hz)")
        self.complex_plane.set_ylabel("-Im[Z] (Ohm)")
        if self.admittance_plot: #This was Bernhard Gadermaier's suggestion. Now modified: no longer real/imaginary conductivity but admittance; this makes it more general.
            self.amplitude.set_title("Real and imaginary admittance")
            self.amplitude.set_ylabel("Y\' ($\Omega^{-1}$)", color = "#114077")
            self.amplitude.tick_params(axis = "y", labelcolor = "#114077")
            self.phase.set_ylabel("Y\'\' ($\Omega^{-1}$)", color = "#117740")
            self.phase.tick_params(axis = "y", labelcolor = "#117740")
            self.phase.yaxis.tick_right()
        else:
            self.amplitude.set_title("Amplitude and phase angle")
            self.amplitude.set_ylabel("|Z| (Ohm)", color = "#114077")
            self.amplitude.tick_params(axis = "y", labelcolor = "#114077")
            self.phase.set_ylabel("Phase angle (degrees)", color = "#117740")
            self.phase.tick_params(axis = "y", labelcolor = "#117740")
            self.phase.yaxis.tick_right()

        #Log-scale the amplitude if requested via self.logamp.
        self.amplitude.set_xscale("log")
        if self.logamp:
            self.amplitude.set_yscale("log")
        if self.logphase:
            self.phase.set_yscale("log")

        #Plot the data in all subplots, if we want it to be visible.
        if self.datavis:
            #Standard complex plane plot
            self.complex_plane.plot(data.real, -data.imag, marker = ".", linestyle = "None", color = "#000000")
            #Frequency labelling
            if len(self.mfreq_freq) > 0:
                self.applyFrequencyMarking()
            #Ghost data in complex plane
            for i in range(min([len(ghost_data_visibility), 3])):
                self.complex_plane.plot(ghost_data[ghost_data_visibility[i]].data.real, -ghost_data[ghost_data_visibility[i]].data.imag, marker = ".", linestyle = "None", color = self.ghost_colours[i])
            #RHS plot style: AB-sigma
            if self.admittance_plot:
                if self.ampvis:
                    self.amplitude.plot(data.freq, 1/data.real, marker = "^", linestyle = "None", color = "#114077")
                if self.phavis:
                    self.phase.plot(data.freq, -1/data.imag, marker = "v", linestyle = "None", color = "#117740")
                #Ghost data RHS
                for i in range(min([len(ghost_data_visibility), 3])):
                    if self.ampvis:
                        self.amplitude.plot(ghost_data[ghost_data_visibility[i]].data.freq, 1/ghost_data[ghost_data_visibility[i]].data.real, marker = "^", linestyle = "None", color = self.ghost_colours[i])
                    if self.phavis:
                        self.phase.plot(ghost_data[ghost_data_visibility[i]].data.freq, -1/ghost_data[ghost_data_visibility[i]].data.imag, marker = "v", linestyle = "None", color = self.ghost_colours[i])
            #RHS plot style: Bode amplitude and phase
            else:
                if self.ampvis:
                    self.amplitude.plot(data.freq, data.amplitude, marker = ".", linestyle = "None", color = "#114077")
                if self.phavis:
                    self.phase.plot(data.freq, (180/np.pi)*data.phase, marker = "s", linestyle = "None", color = "#117740")
                #Ghost data RHS
                for i in range(min([len(ghost_data_visibility), 3])):
                    if self.ampvis:
                        self.amplitude.plot(ghost_data[ghost_data_visibility[i]].data.freq, ghost_data[ghost_data_visibility[i]].data.amplitude, marker = ".", linestyle = "None", color = self.ghost_colours[i])
                    if self.phavis:
                        self.phase.plot(ghost_data[ghost_data_visibility[i]].data.freq, (180/np.pi)*ghost_data[ghost_data_visibility[i]].data.phase, marker = "s", linestyle = "None", color = self.ghost_colours[i])

        #Plot the model in all subplots.
        if self.fitvis and model != None: #...if the model exists, that is. And if we want it to be visible.
            #model.phase, model.amplitude, model.real, model.imag = model.phase[1], model.amplitude[1], model.real[1], model.imag[1] #For some reason, the arrays are not returned cleanly, hence the [1].
            #Plot the model impedance in the complex plane.
            self.complex_plane.plot(model.real, -model.imag, marker = "None", linestyle = "-", color = "#DD4444")
            for i in range(min([len(ghost_data_visibility), 3])):
                self.complex_plane.plot(ghost_data[ghost_data_visibility[i]].model.real, -ghost_data[ghost_data_visibility[i]].model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i])
            #Plot either the real and imaginary conductivities versus frequency, or plot the absolute impedance and the phase against frequency
            if self.admittance_plot:
                if self.ampvis:
                    self.amplitude.plot(model.freq, 1/model.real, marker = "None", linestyle = "-", color = "#4444DD") #Plot the real model admittance against the frequency
                if self.phavis:
                    self.phase.plot(model.freq, -1/model.imag, marker = "None", linestyle = "-", color = "#44DD44") #Plot the imaginary model admittance against the frequency.
                for i in range(min([len(ghost_data_visibility), 3])):
                    if self.ampvis:
                        self.amplitude.plot(ghost_data[ghost_data_visibility[i]].model.freq, 1/ghost_data[ghost_data_visibility[i]].model.real, marker = "None", linestyle = "-", color = self.ghost_m_colours[i]) #Plot the real model admittance against the frequency
                    if self.phavis:
                        self.phase.plot(ghost_data[ghost_data_visibility[i]].model.freq, -1/ghost_data[ghost_data_visibility[i]].model.imag, marker = "None", linestyle = "-", color = self.ghost_m_colours[i]) #Plot the imaginary model admittance against the frequency.
            else:
                if self.ampvis:
                    self.amplitude.plot(model.freq, model.amplitude, marker = "None", linestyle = "-", color = "#4444DD") #Plot the model impedance amplitude against the frequency.
                if self.phavis:
                    self.phase.plot(model.freq, (180/np.pi)*model.phase, marker = "None", linestyle = "-", color = "#44DD44") #Plot the model impedance phase against the frequency.
                for i in range(min([len(ghost_data_visibility), 3])):
                    if self.ampvis:
                        self.amplitude.plot(ghost_data[ghost_data_visibility[i]].model.freq, ghost_data[ghost_data_visibility[i]].model.amplitude, marker = "None", linestyle = "-", color = self.ghost_m_colours[i]) #Plot the model impedance amplitude against the frequency.
                    if self.phavis:
                        self.phase.plot(ghost_data[ghost_data_visibility[i]].model.freq, (180/np.pi)*ghost_data[ghost_data_visibility[i]].model.phase, marker = "None", linestyle = "-", color = self.ghost_m_colours[i]) #Plot the model impedance phase against the frequency.

        #Set the subplots' boundaries.
        self.base_lims.real = (self.d_extend(min(data.real)), self.u_extend(max(data.real)))
        self.base_lims.imag = (self.d_extend(min(-data.imag)), self.u_extend(max(-data.imag)))
        self.base_lims.freq = (self.d_extend(min(data.freq)), self.u_extend(max(data.freq)))
        if self.admittance_plot:
            self.base_lims.amp = (self.d_extend(min(1/data.real)), self.u_extend(max(1/data.real)))
            self.base_lims.phase = (self.d_extend(min(-1/data.imag)), self.u_extend(max(-1/data.imag)))
        else:
            self.base_lims.amp = (self.d_extend(min(data.amplitude)), self.u_extend(max(data.amplitude)))
            self.base_lims.phase = (self.d_extend(min((180/np.pi)*data.phase)), self.u_extend(max((180/np.pi)*data.phase)))
        if self.limiter.enabled: #If the limiter is enabled, use it to limit the subplots' boundaries.
            self.complex_plane.set(xlim = self.limiter.real)
            self.complex_plane.set(ylim = self.limiter.imag)
            self.amplitude.set(xlim = self.limiter.freq)
            self.amplitude.set(ylim = self.limiter.amp)
            self.phase.set(ylim = self.limiter.phase)
        else: #If the limiter is not enabled, apply automatic plot boundaries (defined a few lines above) via base_lims.
            self.complex_plane.set(xlim = self.base_lims.real)
            self.complex_plane.set(ylim = self.base_lims.imag)
            self.amplitude.set(xlim = self.base_lims.freq)
            self.amplitude.set(ylim = self.base_lims.amp)
            self.phase.set(ylim = self.base_lims.phase)

        #Draw the new model, along with the data, on the canvas.
        self.canvas.draw()
