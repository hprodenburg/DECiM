"""Part of DECiM. This file contains operating instructions. Last modified 22 April 2024 by Henrik Rodenburg.

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
        
        Through the 'File' menu, DECiM can load data files (text files) containing whitespace character-separated columns containing 1) frequency, 2) real impedance, and 3) imaginary impedance data. It is possible to change this with the 'Specify data file layout...' option in the same menu.
        After determining the parameters of an equivalent circuit model, the parameters, the raw data, and the model curve can be saved to a .recm2 file. This can be loaded by DECiM and, since it is a text file, it is also human-readable.
        
        If desired, data can be validated using the Z-HIT alogrithm via the 'Perform Z-HIT transform' option in the 'Calculate' menu.
                
        Fitting can be done by selecting an equivalent circuit and then moving the slider, or by refinement. The button at the bottom of the window can be used to select a circuit. The 'Parameter' dropdown can then be used to select a parameter.
        While it is possible to explicitly provide a linear or logarithmic slider response, as well as limits, it is often sufficient to press the 'Adjust controls' button.
        Lastly for manual fitting, is possible to override the slider by using the 'Direct input' field and pressing 'Set', though the slider is often more convenient.
        
        Refinement is possible via two simple refinements that only require one button press, or via the 'Advanced refinement...' option in the 'Calculate' menu.
        The advanced refinement option allows parameters to be fixed and the frequency range to be varied. It is more powerful and versatile than the simple refinement -- the simple refinement is a special case of the advanced refinement in which all parameters are optimized with unit weighting.
        
        The 'Plot' menu contains cosmetic options. Perhaps the most important option here is 'Reset view', which works better than the 'Home' button below the plot canvas.
        
        Lastly, the 'History' menu allows multiple data sets to be stored in RAM. It is also possible to plot multiple spectra for comparison through this menu.
        
        Last updated 22 April 2024 by Henrik Rodenburg."""
        
        #Cut the help string into lines that do not run off the screen.
        line_length = 0
        new_help_info = ""
        for h in range(len(self.help_info)):
            if self.help_info[h] == "\n" or (line_length > 200 and self.help_info[h] == " "):
                line_length = 0
                new_help_info += "\n"
            elif self.help_info[h] == " " and self.help_info[h-1] in ["\n", " "]:
                continue
            else:
                new_help_info += self.help_info[h]
                line_length += 1
        self.help_info = new_help_info
        
        #Display the information
        self.help_label = tk.Label(self, text = self.help_info, justify = tk.LEFT)
        self.help_label.pack(side = tk.TOP, anchor = tk.W, fill = tk.BOTH)
        
        self.exit_button = tk.Button(self, text = "OK", command = self.destroy)
        self.exit_button.pack(side = tk.TOP, anchor = tk.CENTER)