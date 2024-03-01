"""Part of DECiM. This file contains operating instructions. Last modified 6 December 2023 by Henrik Rodenburg.

Classes:
HelpWindow -- displays a short program manual"""

###########
##IMPORTS##
###########

import tkinter as tk
import tkinter.ttk as ttk

###############
##HELP WINDOW##
###############

class HelpWindow(tk.Toplevel):
    def __init__(self):
        """Window to display a short program manual.
        
        Attributes:
        width -- window width
        height -- window height
        
        help_info -- short manual
        help_label -- label to display help_info
        exit_button -- button to close the window"""
        super().__init__()
        self.title("Summary of DECiM operation")
        self.width = 1280
        self.height = 400
        self.geometry("{:d}x{:d}".format(self.width, self.height))
    
        self.help_info = """DECiM (Determination of Equivalent Circuit Models) is a program written for the analysis of impedance spectra. It can be used to fit simple equivalent circuit models to spectra.
        
        Through the \'File\' menu, DECiM can load data files (text files) containing whitespace character-separated columns containing 1) frequency, 2) real impedance, and 3) imaginary impedance data.
        After determining the parameters of an equivalent circuit model, the parameters, the raw data, and the model curve can be saved to a .recm2 file. This can be loaded by DECiM and, since it is a text file, it is also human-readable.
                
        Fitting can be done by selecting an equivalent circuit and then moving the slider, or by refinement. The button at the bottom of the window can be used to select a circuit. The \'Parameter\' dropdown can then be used to select a parameter.
        While it is possible to explicitly provide a linear or logarithmic slider response, as well as limits, it is often sufficient to press the \'Adjust controls\' button.
        Lastly for manual fitting, is possible to override the slider by using the \'Direct input\' field and pressing \'Set\', though the slider is often more convenient.
        
        Refinement is possible via two simple refinements that only require one button press, or via the \'Advanced refinement...\' option in the \'Calculate\' menu.
        The advanced refinement option allows parameters to be fixed and the frequency range to be varied. It is more powerful and versatile than the simple refinement -- the simple refinement is a special case of the advanced refinement in which all parameters are optimized with unit weighting.
        
        The \'Plot\' menu contains cosmetic options. Perhaps the most important option here is \'Reset view\', which works better than the \'Home\' button below the plot canvas.
        
        Lastly, the \'History\' menu allows multiple data sets to be stored in RAM. It is also possible to plot multiple spectra for comparison through this menu.
        
        Last updated 4 December 2023 by Henrik Rodenburg."""
        
        self.help_label = tk.Label(self, text = self.help_info)
        self.help_label.pack(side = tk.TOP, anchor = tk.W)
        
        self.exit_button = tk.Button(self, text = "OK", command = self.destroy)
        self.exit_button.pack(side = tk.TOP, anchor = tk.CENTER)