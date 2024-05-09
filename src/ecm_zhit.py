"""Part of DECiM. This file contains the Z-HIT maths and window. Last modified 9 May 2024 by Henrik Rodenburg.

Classes:
ZHITEngine -- handles the Z-HIT transform
ZHITWindow -- builds the GUI around the Z-HIT transform as implemented in ZHITEngine"""

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
import scipy.signal as sg
import copy

import tkinter as tk
import tkinter.ttk as ttk

from ecm_datastructure import dataSet
from ecm_helpers import nearest

################
##Z-HIT WINDOW##
################

class ZHITEngine():
    def __init__(self, data, high_lim = 1e6, low_lim = 1e-2, cut_hf_edge = True, cut_lf_edge = False, savgol_winlength = 20, savgol_polyorder = 2):
        """Z-HIT transform engine. Performs all the numerical operations in the Z-HIT transform.
        
        Init arguments, becoming attributes under the same name:
        
        data -- dataSet of measured data
        low_lim -- frequency lower limit
        high_lim -- frequency upper limit
        
        cut_hf_edge -- Boolean; cut the high-frequency smoothed data
        cut_lf_edge -- Boolean; cut the low-frequency smoothed data
        
        savgol_winlength -- window length for filter
        savgol_polyorder -- polynomial_order for filter
        
        Further starting attributes:
        high_cut -- number of points to cut from self.lnZ, self.transformed_data.amplitude, self.transformed_data.freq, and self.transformed_data.phase if cut_hf_edge; normally 1/10th of data length
        low_cut -- number of points to cut from self.lnZ, self.transformed_data.amplitude, self.transformed_data.freq, and self.transformed_data.phase if cut_lf_edge; normally 1/10th of data length
        
        Attributes created during transform calculation:
        
        lnZ -- ln|Z| calculated from phase
        iconstant -- NumPy array containing one value, the integration constant C that best corrects the offset in |Z|
        transformed_data -- dataSet of transformed data
        backup_data -- dataSet to recover initial data
        
        Methods:
        
        smooth_phase -- Use a Savitzky-Golay filter to smoothen the phase data
        cut_edges -- remove the ends of the smoothed phase, and other arrays; works well against edge artifacts
        gamma -- helper function related to the Riemann zeta function
        integrate -- integral calculation
        derivative -- first derivative calculation
        raw_lnZ -- compute ln|Z| with unknown integration constant
        error_lnZ -- sum of the squares of the differences between the measured |Z| and newly calculated ln|Z|
        determine_integration_constant -- correct |Z| by determining the integration constant
        phase_to_original_dimensions -- reduce the number of phase data points to the measured number
        compute_zhit_transform -- compute the Z-HIT transformed data and save it to a new dataSet"""
        self.data = data
        self.transformed_data = copy.deepcopy(data)
        self.backup_data = copy.deepcopy(data)
        self.low_lim = low_lim
        self.high_lim = high_lim
        self.low_cut = int(len(data.freq)/10)
        self.high_cut = int(len(data.freq)/10)
        self.cut_hf_edge = cut_hf_edge
        self.cut_lf_edge = cut_lf_edge
        self.savgol_winlength = savgol_winlength
        self.savgol_polyorder = savgol_polyorder
                
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
        return x, sg.savgol_filter(y, self.savgol_winlength, self.savgol_polyorder, deriv = 1)
        
    def smooth_phase(self):
        """Use a Savitzky-Golay filter to smoothen the phase."""
        self.transformed_data.phase = sg.savgol_filter(self.transformed_data.phase, self.savgol_winlength, self.savgol_polyorder, deriv = 0)
        
    def cut_edges(self):
        """Remove the poorly smoothed edges of the phase data for the derivative calculation."""
        if self.cut_hf_edge and max(self.transformed_data.freq) == self.transformed_data.freq[-1]:
            self.transformed_data.freq = self.transformed_data.freq[:-self.high_cut]
            self.transformed_data.phase = self.transformed_data.phase[:-self.high_cut]
            self.transformed_data.amplitude = self.transformed_data.amplitude[:-self.high_cut]
        elif self.cut_hf_edge and max(self.transformed_data.freq) == self.transformed_data.freq[0]:
            self.transformed_data.freq = self.transformed_data.freq[self.high_cut:]
            self.transformed_data.phase = self.transformed_data.phase[self.high_cut:]
            self.transformed_data.amplitude = self.transformed_data.amplitude[self.high_cut:]
        if self.cut_lf_edge and min(self.transformed_data.freq) == self.transformed_data.freq[-1]:
            self.transformed_data.freq = self.transformed_data.freq[:-self.low_cut]
            self.transformed_data.phase = self.transformed_data.phase[:-self.low_cut]
            self.transformed_data.amplitude = self.transformed_data.amplitude[:-self.low_cut]
        elif self.cut_lf_edge and min(self.transformed_data.freq) == self.transformed_data.freq[0]:
            self.transformed_data.freq = self.transformed_data.freq[self.low_cut:]
            self.transformed_data.phase = self.transformed_data.phase[self.low_cut:]
            self.transformed_data.amplitude = self.transformed_data.amplitude[self.low_cut:]
        
    def raw_lnZ(self):
        """Calculate ln|Z| following the Z-HIT transform procedure."""
        lnw = np.log(2*np.pi*self.transformed_data.freq)
        integrals = []
        for i in range(len(lnw) + 1):
            integrals.append(self.integrate(lnw[:i], self.transformed_data.phase[:i]))
        integrals = np.array(integrals)
        self.dlnw, self.dphase = self.derivative(lnw, self.transformed_data.phase)
        self.lnZ = (2/np.pi)*integrals[:-1] + self.gamma(1)*self.dphase #Constant of integration is omitted here
        
    def error_lnZ(self, fpars):
        """Calculate the sum of the squares of the differences between the measured and calculated ln|Z|.
        
        Arguments:
        self
        fpars -- array containing only one parameter: C, the integration constant
        
        Returns:
        Sum of the squares of the differences between the measured and calculated ln|Z|"""
        return sum((np.log(self.transformed_data.amplitude[nearest(self.low_lim, self.transformed_data.freq):nearest(self.high_lim, self.transformed_data.freq) + 1]) - (self.lnZ + fpars[0]))**2)
        
    def determine_integration_constant(self):
        """Determine the integration constant C in the Z-HIT transform procedure."""
        self.iconstant = np.zeros(1)
        opt_res = op.minimize(self.error_lnZ, self.iconstant, method = "Nelder-Mead", options = {"maxiter": 500})
        self.iconstant = opt_res["x"]
        
    def compute_zhit_transform(self):
        self.data = copy.deepcopy(self.backup_data)
        self.transformed_data.freq = self.transformed_data.freq[nearest(self.low_lim, self.data.freq):nearest(self.high_lim, self.data.freq)] #Frequency limits
        self.transformed_data.real = self.transformed_data.real[nearest(self.low_lim, self.data.freq):nearest(self.high_lim, self.data.freq)]
        self.transformed_data.imag = self.transformed_data.imag[nearest(self.low_lim, self.data.freq):nearest(self.high_lim, self.data.freq)]
        self.transformed_data.amplitude = self.transformed_data.amplitude[nearest(self.low_lim, self.data.freq):nearest(self.high_lim, self.data.freq)]
        self.transformed_data.phase = self.transformed_data.phase[nearest(self.low_lim, self.data.freq):nearest(self.high_lim, self.data.freq)]
        self.smooth_phase() #New phase
        self.cut_edges()
        self.raw_lnZ() #Towards new impedance;
        self.determine_integration_constant() #Again, towards new impedance;
        self.transformed_data.amplitude = np.exp(self.lnZ + self.iconstant) #New impedance
        #self.transformed_data.freq = self.data.freq[nearest(self.low_lim, self.data.freq):nearest(self.high_lim, self.data.freq) + 1]
        self.transformed_data.real = self.transformed_data.amplitude*np.cos(self.transformed_data.phase)
        self.transformed_data.imag = self.transformed_data.amplitude*np.sin(self.transformed_data.phase)

class ZHITWindow(tk.Toplevel):
    def __init__(self, data):
        """Z-HIT analysis window. Builds the GUI around the ZHITEngine.
        
        Init arguments:
        data -- dataSet of measured data
        
        Attributes (except UI elements):
        width -- window width
        height -- window height
        
        raw_data -- dataSet of measured data (deep copy)
        data -- dataSet of measured data (deep copy)
        
        plot_zhit_result -- Boolean, indicates if result can be plotted
        correction_accepted -- Boolean, indicates if data transformation is accepted
        
        show_derivative -- debugging option, crudely plots the derivative of the phase
        
        low_lim -- frequency lower limit
        high_lim -- frequency upper limit
        
        Methods:
        make_UI -- make the UI
        make_plotting_frame -- make the ttk.Frame on which the plot canvas is placed
        
        update_settings -- update the settings for the Z-HIT transform procedure
        calculate_zhit_amplitude -- allow the Z-HIT transform to be calculated by the update_plot_canvas method
        reject_and_close -- close the window without accepting the data transformation
        accept_and_close -- accept the data transformation and close the window
        toggle_filter_entries -- enable or disable Savitzky-Golay filter tk.Entries and tk.Labels
        
        update_plot_canvas -- clear and redraw the plot canvas; also includes the whole Z-HIT procedure"""
        super().__init__()
        self.title("Z-HIT transform")
        self.width = int(self.winfo_screenwidth()*0.85)
        self.height = int(self.winfo_screenheight()*0.85)
        self.geometry("{:d}x{:d}".format(self.width, self.height))
        self.raw_data = data
        self.data = copy.deepcopy(data)
        self.low_lim = min(data.freq)
        self.high_lim = max(data.freq)
        self.window_length = 50
        self.polynomial_order = 3
        self.plot_zhit_result = False
        self.correction_accepted = False
        self.show_derivative = False
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
        
        #Entry boxes for Z-HIT frequency range
        self.settings_frame = ttk.Frame(self.controls_frame)
        self.settings_frame.pack(side = tk.TOP, anchor = tk.CENTER)
        self.low_freq_frame = ttk.Frame(self.settings_frame)
        self.low_freq_frame.pack(side = tk.LEFT, anchor = tk.N)
        self.high_freq_frame = ttk.Frame(self.settings_frame)
        self.high_freq_frame.pack(side = tk.LEFT, anchor = tk.N)
        self.cut_frame = ttk.Frame(self.settings_frame)
        self.cut_frame.pack(side = tk.LEFT, anchor = tk.S)
        self.freq_label = tk.Label(self.low_freq_frame, text = "Frequency range:")
        self.freq_label.pack(side = tk.LEFT, anchor = tk.CENTER, padx = 5)
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
        self.cut_label = tk.Label(self.cut_frame, text = "Cut edge artifacts:")
        self.cut_label.pack(side = tk.LEFT, anchor = tk.CENTER, padx = 5)
        self.low_f_cut = tk.IntVar()
        self.low_cut_tickbox = tk.Checkbutton(self.cut_frame, text = "Low f", variable = self.low_f_cut, onvalue = 1, offvalue = 0)
        self.low_cut_tickbox.pack(side = tk.LEFT, anchor = tk.CENTER)
        self.high_f_cut = tk.IntVar()
        self.high_cut_tickbox = tk.Checkbutton(self.cut_frame, text = "High f", variable = self.high_f_cut, onvalue = 1, offvalue = 0)
        self.high_cut_tickbox.pack(side = tk.LEFT, anchor = tk.CENTER)
        
        #Buttons
        self.buttons_frame = ttk.Frame(self.controls_frame)
        self.buttons_frame.pack(side = tk.TOP, anchor = tk.CENTER, pady = 10)
        self.calculate_button = tk.Button(self.buttons_frame, text = "Calculate Z-HIT impedance", command = self.calculate_zhit_amplitude)
        self.calculate_button.pack(side = tk.RIGHT, anchor = tk.CENTER)
        self.reject_button = tk.Button(self.buttons_frame, text = "Reject and close", command = self.reject_and_close)
        self.reject_button.pack(side = tk.RIGHT, anchor = tk.CENTER)
        self.accept_button = tk.Button(self.buttons_frame, text = "Accept, save and close", command = self.accept_and_close)
        self.accept_button.pack(side = tk.RIGHT, anchor = tk.CENTER)
        
        #Savitzky-Golay filter settings
        self.filter_frame = ttk.Frame(self.controls_frame)
        self.filter_frame.pack(side = tk.TOP, anchor = tk.CENTER)
        self.filter_label = tk.Label(self.filter_frame, text = "Savitzky-Golay filter:")
        self.filter_label.pack(side = tk.LEFT, anchor = tk.CENTER, padx = 5)
        self.filter_winlength = tk.StringVar()
        self.filter_winlength.set(str(self.window_length))
        self.filter_polyorder = tk.StringVar()
        self.filter_polyorder.set(str(self.polynomial_order))
        self.winlength_label = tk.Label(self.filter_frame, text = "Window length:")
        self.winlength_label.pack(side = tk.LEFT, anchor = tk.CENTER)
        self.winlength_entry = tk.Entry(self.filter_frame, textvariable = self.filter_winlength)
        self.winlength_entry.pack(side = tk.LEFT, anchor = tk.CENTER)
        self.polyorder_label = tk.Label(self.filter_frame, text = "Polynomial order:")
        self.polyorder_label.pack(side = tk.LEFT, anchor = tk.CENTER)
        self.polyorder_entry = tk.Entry(self.filter_frame, textvariable = self.filter_polyorder)
        self.polyorder_entry.pack(side = tk.LEFT, anchor = tk.CENTER)
        
    def make_plotting_frame(self):
        """Make the plot canvas, put the plots on it and create the toolbar."""
        #Figure initialisation
        self.fig, (self.nyquist, self.bode_amp) = pt.subplots(nrows = 1, ncols = 2)
        self.bode_phase = self.bode_amp.twinx()
        self.bode_phase.yaxis.tick_right()
        self.bode_phase.yaxis.set_label_position("right")
        self.canvas = btk.FigureCanvasTkAgg(self.fig, self.plotting_frame)
        self.canvas.get_tk_widget().pack(side = tk.TOP, fill = tk.BOTH, expand = True)
        #Toolbar
        self.toolbar = btk.NavigationToolbar2Tk(self.canvas, self.plotting_frame)
        self.toolbar.update()
        #Labels and plotting
        self.update_plot_canvas()
        
    #Commands from UI
    def update_settings(self):
        """Update the frequency limits and the parameters used to create the spline curve."""
        self.low_lim = float(self.low_freq_par.get())
        self.high_lim = float(self.high_freq_par.get())
        self.window_length = int(self.winlength_entry.get())
        self.polynomial_order = int(self.polyorder_entry.get())
    
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
        self.bode_phase.yaxis.tick_right()
        self.bode_phase.yaxis.set_label_position("right")
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
            #Get the Savitzky-Golay filter options
            bool_dict = {0: False, 1: True}
            zhit_engine = ZHITEngine(self.raw_data, low_lim = self.low_lim, high_lim = self.high_lim, cut_lf_edge = bool_dict[self.low_f_cut.get()], cut_hf_edge = bool_dict[self.high_f_cut.get()], savgol_winlength = self.window_length, savgol_polyorder = self.polynomial_order) #Start a new ZHITEngine
            zhit_engine.compute_zhit_transform() #Compute the Z-HIT transform
            self.nyquist.plot(zhit_engine.transformed_data.real, -zhit_engine.transformed_data.imag, linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444") #Replot everything
            self.bode_amp.plot(zhit_engine.transformed_data.freq, zhit_engine.transformed_data.amplitude, linestyle = "-", marker = "None", linewidth = 1.5, color = "#4444DD")
            self.bode_phase.plot(zhit_engine.transformed_data.freq, zhit_engine.transformed_data.phase*180/np.pi, linestyle = "-", marker = "None", linewidth = 1.5, color = "#44DD44", label = "$\phi$ ($^\circ$)")
            if self.show_derivative:
                self.bode_phase.plot(zhit_engine.transformed_data.freq, zhit_engine.dphase*180/np.pi, linestyle = "-", marker = "None", linewidth = 1.5, color = "#DD4444", label = "d$\phi$/dln$\omega$ ($^\circ$/ln(Hz))")
                self.bode_phase.legend(loc = "upper right", frameon = False, labelcolor = "linecolor", labelspacing = .05, handlelength = 1.25, handletextpad = .5)
            self.data = zhit_engine.transformed_data #Save the transformed data
        self.canvas.draw()
        