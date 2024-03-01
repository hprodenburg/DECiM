"""Part of DECiM. This file contains the Z-HIT maths and window. Last modified 9 January 2024 by Henrik Rodenburg.

Classes:
ZHITWindow -- handles the Z-HIT transform"""

###########
##IMPORTS##
###########

import numpy as np

import matplotlib as mp
import matplotlib.pyplot as pt
import matplotlib.backends.backend_tkagg as btk
import matplotlib.figure as fg
import matplotlib.animation as anim

import scipy.optimize as op
import scipy.interpolate as ir
import copy

import tkinter as tk
import tkinter.ttk as ttk

from ecm_datastructure import dataSet
from ecm_helpers import nearest

################
##Z-HIT WINDOW##
################

class ZHITWindow(tk.Toplevel):
    def __init__(self, data):
        """Z-HIT analysis window.
        
        Init arguments:
        data -- dataSet of measured data
        
        Attributes (except UI elements):
        width -- window width
        height -- window height
        
        raw_data -- dataSet of measured data (deep copy)
        data -- dataSet of measured data (deep copy)
        
        plot_zhit_result -- Boolean, indicates if result can be plotted
        correction_accepted -- Boolean, indicates if data transformation is accepted
        
        low_lim -- frequency lower limit
        high_lim -- frequency upper limit
        
        smooth_n -- number of frequency points for the spline relative to the original data
        smooth_s -- spline smoothness parameter, the 's' argument in ir.UnivariateSpline
        
        Methods:
        make_UI -- make the UI
        make_plotting_frame -- make the ttk.Frame on which the plot canvas is placed
        
        update_settings -- update the settings for the Z-HIT transform procedure
        calculate_zhit_amplitude -- allow the Z-HIT transform to be calculated by the update_plot_canvas method
        reject_and_close -- close the window without accepting the data transformation
        accept_and_close -- accept the data transformation and close the window
        
        update_plot_canvas -- clear and redraw the plot canvas; also includes the whole Z-HIT procedure
        
        spline_interpolation -- interpolate the phase
        gamma -- helper function related to the Riemann zeta function
        integrate -- integral calculation
        derivative -- first derivative calculation
        raw_lnZ -- compute ln|Z| with unknown integration constant
        error_lnZ -- sum of the squares of the differences between the measured |Z| and newly calculated ln|Z|
        determine_integration_constant -- correct |Z| by determining the integration constant
        lnZ_to_original_dimensions -- reduce the number of |Z| data points to the measured number
        phase_to_original_dimensions -- reduce the number of phase data points to the measured number"""
        super().__init__()
        self.title("Z-HIT transform")
        self.width = int(self.winfo_screenwidth()*0.85)
        self.height = int(self.winfo_screenheight()*0.85)
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        self.raw_data = copy.deepcopy(data)
        self.data = copy.deepcopy(data)
        self.plot_zhit_result = False
        self.correction_accepted = False
        self.low_lim = min(self.data.freq)
        self.high_lim = max(self.data.freq)
        self.smooth_n = 10
        self.smooth_s = 0.03
        self.make_UI()
        
    #UI initialisation
    def make_UI(self):
        """Make a high and low ttk.Frame, then fill the lower frame with new frames for the settings tk.Entries and tk.Labels. Then, add buttons onto the lower frame."""
        #Two frames for vertical division
        self.plotting_frame = ttk.Frame(self) #This is where the plot canvas goes
        self.plotting_frame.pack(side = tk.TOP, anchor = tk.CENTER, fill = tk.BOTH, expand = tk.YES)
        self.controls_frame = ttk.Frame(self) #This is where the accept and reject buttons go
        self.controls_frame.pack(side = tk.TOP, anchor = tk.CENTER, fill = tk.BOTH, expand = tk.YES)
        
        #Making the plotting frame
        self.make_plotting_frame()
        
        #Entry boxes for Z-HIT frequency range and smoothing parameters
        self.settings_frame = ttk.Frame(self.controls_frame)
        self.settings_frame.pack(side = tk.TOP, anchor = tk.CENTER)
        self.low_freq_frame = ttk.Frame(self.settings_frame)
        self.low_freq_frame.pack(side = tk.LEFT, anchor = tk.N)
        self.high_freq_frame = ttk.Frame(self.settings_frame)
        self.high_freq_frame.pack(side = tk.LEFT, anchor = tk.N)
        self.smooth_n_frame = ttk.Frame(self.settings_frame)
        self.smooth_n_frame.pack(side = tk.LEFT, anchor = tk.N)
        self.smooth_s_frame = ttk.Frame(self.settings_frame)
        self.smooth_s_frame.pack(side = tk.LEFT, anchor = tk.N)
        self.low_freq_par = tk.StringVar()
        self.low_freq_par.set(str(self.low_lim))
        self.low_freq_label = tk.Label(self.low_freq_frame, text = "Lowest\nfrequency (Hz)")
        self.low_freq_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.low_freq_entry = tk.Entry(self.low_freq_frame, textvariable = self.low_freq_par)
        self.low_freq_entry.pack(side = tk.TOP, anchor = tk.CENTER)
        self.high_freq_par = tk.StringVar()
        self.high_freq_par.set(str(self.high_lim))
        self.high_freq_label = tk.Label(self.high_freq_frame, text = "Highest\nfrequency (Hz)")
        self.high_freq_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.high_freq_entry = tk.Entry(self.high_freq_frame, textvariable = self.high_freq_par)
        self.high_freq_entry.pack(side = tk.TOP, anchor = tk.CENTER)
        self.smooth_n_par = tk.StringVar()
        self.smooth_n_par.set(str(self.smooth_n))
        self.smooth_n_label = tk.Label(self.smooth_n_frame, text = "No. of points: spline\nvs. original data")
        self.smooth_n_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.smooth_n_entry = tk.Entry(self.smooth_n_frame, textvariable = self.smooth_n_par)
        self.smooth_n_entry.pack(side = tk.TOP, anchor = tk.CENTER)
        self.smooth_s_par = tk.StringVar()
        self.smooth_s_par.set(str(self.smooth_s))
        self.smooth_s_label = tk.Label(self.smooth_s_frame, text = "Smoothness\nof spline fit")
        self.smooth_s_label.pack(side = tk.TOP, anchor = tk.CENTER)
        self.smooth_s_entry = tk.Entry(self.smooth_s_frame, textvariable = self.smooth_s_par)
        self.smooth_s_entry.pack(side = tk.TOP, anchor = tk.CENTER)
        
        #Buttons
        self.buttons_frame = ttk.Frame(self.controls_frame)
        self.buttons_frame.pack(side = tk.TOP, anchor = tk.CENTER, pady = 25)
        self.calculate_button = tk.Button(self.buttons_frame, text = "Calculate Z-HIT impedance", command = self.calculate_zhit_amplitude)
        self.calculate_button.pack(side = tk.RIGHT, anchor = tk.CENTER)
        self.reject_button = tk.Button(self.buttons_frame, text = "Reject and close", command = self.reject_and_close)
        self.reject_button.pack(side = tk.RIGHT, anchor = tk.CENTER)
        self.accept_button = tk.Button(self.buttons_frame, text = "Accept, save and close", command = self.accept_and_close)
        self.accept_button.pack(side = tk.RIGHT, anchor = tk.CENTER)
        
    def make_plotting_frame(self):
        """Make the plot canvas, put the plots on it and create the toolbar."""
        #Figure initialisation
        self.fig, (self.nyquist, self.bode_amp) = pt.subplots(nrows = 1, ncols = 2)
        self.bode_phase = self.bode_amp.twinx()
        self.canvas = btk.FigureCanvasTkAgg(self.fig, self.plotting_frame)
        self.canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        #Toolbar
        self.toolbar = btk.NavigationToolbar2Tk(self.canvas, self.plotting_frame)
        self.toolbar.update()
        #Labels and plotting
        self.update_plot_canvas()
        
    #Commands from UI
    def update_settings(self):
        """Üpdate the frequency limits and the parameters used to create the spline curve."""
        self.low_lim = float(self.low_freq_par.get())
        self.high_lim = float(self.high_freq_par.get())
        self.smooth_n = int(self.smooth_n_par.get())
        self.smooth_s = float(self.smooth_s_par.get())
    
    def calculate_zhit_amplitude(self):
        """Update the settings, allow the Z-HIT amplitude to be calculated and update the plots with update_plot_canvas, which will perform the calculation."""
        self.update_settings()
        self.plot_zhit_result = True
        self.update_plot_canvas()
    
    def reject_and_close(self):
        """Reject the Z-HIT data transformation and close the window."""
        self.correction_accepted = False
        self.destroy()
        
    def accept_and_close(self):
        """Accept the Z-HIT data transformation and close the window."""
        self.correction_accepted = True
        self.destroy()
        
    def update_plot_canvas(self):
        """Clear the plots, redraw them and, if allowed, calculate and plot the Z-HIT result. Then redraw the canvas."""
        #Clear subplots
        self.nyquist.cla()
        self.bode_amp.cla()
        self.bode_phase.cla()
        #(Re)write labels
        self.nyquist.set_xlabel("Re[Z] ($\Omega$)")
        self.nyquist.set_ylabel("-Im[Z] ($\Omega$)")
        self.bode_amp.set_xlabel("f (Hz)")
        self.bode_amp.set_ylabel("|Z| ($\Omega$)")
        self.bode_phase.set_ylabel("$\phi$ ($^\circ$)")
        #Set scales
        self.nyquist.set_xscale("linear")
        self.nyquist.set_yscale("linear")
        self.bode_amp.set_xscale("log")
        self.bode_amp.set_yscale("log")
        self.bode_phase.set_yscale("linear")
        #Axis limits
        self.nyquist.set_xlim(left = min(self.raw_data.real) - 0.05*max(self.raw_data.real), right = 1.05*max(self.raw_data.real))
        self.nyquist.set_ylim(bottom = min(-self.raw_data.imag) - 0.05*max(-self.raw_data.imag), top = 1.05*max(-self.raw_data.imag))
        self.bode_amp.set_xlim(left = min(self.raw_data.freq), right = max(self.raw_data.freq))
        self.bode_amp.set_ylim(bottom = 0.90*min(self.raw_data.amplitude), top = 1.11*max(self.raw_data.amplitude))
        self.bode_phase.set_ylim(bottom = min([1.05*min(self.raw_data.phase), min(self.raw_data.phase) - 0.05*max(self.raw_data.phase)])*180/np.pi, top = max([1.05*max(self.raw_data.phase), max(self.raw_data.phase) - 0.05*min(self.raw_data.phase)])*180/np.pi)
        #Label colours
        self.bode_amp.yaxis.label.set_color("#2222A0")
        self.bode_amp.tick_params(axis = "y", colors = "#2222A0")
        self.bode_phase.yaxis.label.set_color("#229950")
        self.bode_phase.tick_params(axis = "y", colors = "#229950")
        #Raw data plotting
        self.nyquist.plot(self.raw_data.real, -self.raw_data.imag, linestyle = "None", marker = ".", markersize = 6, fillstyle = "full", color = "#A0A0A0", markeredgecolor = "#000000", markeredgewidth = 1)
        self.bode_amp.plot(self.raw_data.freq, self.raw_data.amplitude, linestyle = "None", marker = ".", markersize = 6, fillstyle = "full", color = "#2222A0", markeredgecolor = "#000000", markeredgewidth = 1)
        self.bode_phase.plot(self.raw_data.freq, self.raw_data.phase*180/np.pi, linestyle = "None", marker = "s", markersize = 6, fillstyle = "full", color = "#229950", markeredgecolor = "#000000", markeredgewidth = 1)
        #Z-HIT calculation
        if self.plot_zhit_result:
            self.data.freq = self.raw_data.freq[nearest(self.low_lim, self.raw_data.freq):nearest(self.high_lim, self.raw_data.freq)] #Frequency limits
            self.data.real = self.raw_data.real[nearest(self.low_lim, self.raw_data.freq):nearest(self.high_lim, self.raw_data.freq)]
            self.data.imag = self.raw_data.imag[nearest(self.low_lim, self.raw_data.freq):nearest(self.high_lim, self.raw_data.freq)]
            self.data.amplitude = self.raw_data.amplitude[nearest(self.low_lim, self.raw_data.freq):nearest(self.high_lim, self.raw_data.freq)]
            self.data.phase = self.raw_data.phase[nearest(self.low_lim, self.raw_data.freq):nearest(self.high_lim, self.raw_data.freq)]
            self.spline_interpolation() #New frequency & phase
            self.raw_lnZ() #Towards new impedance...
            self.lnZ_to_original_dimensions()
            self.phase_to_original_dimensions()
            self.determine_integration_constant() #Again, towards new impedance...
            new_Z = np.exp(self.small_lnZ + self.iconstant) #New impedance
            self.data.freq = self.raw_data.freq[nearest(self.low_lim, self.raw_data.freq):nearest(self.high_lim, self.raw_data.freq)]
            self.data.amplitude = new_Z
            self.data.real = new_Z*np.cos(self.data.phase)
            self.data.imag = new_Z*np.sin(self.data.phase)
            self.nyquist.plot(self.data.real, -self.data.imag, linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444") #Replot everything
            self.bode_amp.plot(self.data.freq, self.data.amplitude, linestyle = "-", marker = "None", linewidth = 1.5, color = "#4444DD")
            self.bode_phase.plot(self.data.freq, self.data.phase*180/np.pi, linestyle = "-", marker = "None", linewidth = 1.5, color = "#44DD44")
        self.canvas.draw()
        
    def spline_interpolation(self):
        """Smoothen the phase data with spline interpolation. This will generate a new frequency and phase in data."""
        phase_spline = ir.UnivariateSpline(self.data.freq, self.data.phase, s = self.smooth_s)
        new_freq = 10**np.linspace(np.log10(self.data.freq[0]), np.log10(self.data.freq[-1]), len(self.data.freq)*self.smooth_n)
        new_phase = phase_spline(new_freq)
        self.data.freq = new_freq
        self.data.phase = new_phase
        
    #Mathematics behind the Z-HIT transform
    def gamma(self, k):
        """Compute gamma (part of the Riemann zeta function).
        
        Arguments:
        self
        k -- integer
        
        Returns:
        gamma -- float"""
        zeta = {2: np.pi**2/6, 4: np.pi**4/90, 6: np.pi**6/945, 8: np.pi**8/9450}
        return (-1)**k * (np.pi/2) * (1/2**k) * zeta[k + 1]
        
    def integrate(self, x, y):
        """Integrate data y(x).
        
        Arguments:
        x -- x data
        y -- y data
        
        Returns:
        Integral (scalar)"""
        out = []
        for i in range(len(y) - 1):
            out.append(((y[i] + y[i+1])/2) * (x[i+1] - x[i]))
        return sum(out)
        
    def derivative(self, x, y):
        """Differentiate data y(x)
        
        Arguments:
        x -- x data
        y -- y data
        
        Returns: tuple of:
        new_x -- NumPy array
        new_y -- NumPy array, first derivative of y"""
        new_x, new_y = [], []
        for i in range(len(y) - 1):
            new_x.append((x[i] + x[i+1])/2)
            new_y.append((y[i+1] - y[i])/(x[i+1] - x[i]))
        return np.array(new_x), np.array(new_y)
        
    def raw_lnZ(self):
        """Calculate ln|Z| following the Z-HIT transform procedure."""
        lnw = np.log(2*np.pi*self.data.freq)
        integrals = []
        for i in range(len(lnw) + 1):
            integrals.append(self.integrate(lnw[:i], self.data.phase[:i]))
        integrals = np.array(integrals)
        dlnw, dphase = self.derivative(lnw, self.data.phase)
        self.lnZ = (2/np.pi)*integrals[1:-1] + self.gamma(1)*dphase #Constant of integration is omitted here
        
    def lnZ_to_original_dimensions(self):
        """Reduce the number of ln|Z| data points to the number that was originally measured."""
        small_lnZ = []
        for i in range(0, len(self.lnZ), self.smooth_n):
            small_lnZ.append(sum(self.lnZ[i:i + self.smooth_n])/self.smooth_n)
        self.small_lnZ = np.array(small_lnZ)
        
    def phase_to_original_dimensions(self):
        """Reduce the number of phase data points to the number that was originally measured."""
        small_phase = []
        for i in range(0, len(self.data.phase), self.smooth_n):
            small_phase.append(sum(self.data.phase[i:i + self.smooth_n])/self.smooth_n)
        self.data.phase = np.array(small_phase)
        
    def error_lnZ(self, fpars):
        """Calculate the sum of the squares of the differences between the measured and calculated ln|Z|.
        
        Arguments:
        self
        fpars -- array containing only one parameter: C, the integration constant
        
        Returns:
        Sum of the squares of the differences between the measured and calculated ln|Z|"""
        return sum((np.log(self.data.amplitude) - (self.small_lnZ + fpars[0]))**2)
        
    def determine_integration_constant(self):
        """Determine the integration constant C in the Z-HIT transform procedure."""
        self.iconstant = np.zeros(1)
        opt_res = op.minimize(self.error_lnZ, self.iconstant, method = "Nelder-Mead", options = {"maxiter": 500})
        self.iconstant = opt_res["x"]