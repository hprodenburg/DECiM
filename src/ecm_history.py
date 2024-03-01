"""Part of DECiM. The code in this file deals with the history menu. Last modified 1 March 2024 by Henrik Rodenburg.

Classes:
expandedDataSet -- object holding measurement and model dataSets, a Circuit and list of fit parameters
HistoryManager -- for keeping track of all saved measurements
DataSetSelectorWindow -- for selecting different or additional expandedDataSets for plotting"""

###########
##IMPORTS##
###########

import numpy as np

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.simpledialog as sdiag
import tkinter.filedialog as fdiag

from ecm_datastructure import dataSet

############################
##FULL DATASET INFORMATION##
############################

class expandedDataSet():
    def __init__(self, name, measured_data, model_data, circuit, parameters = list(np.zeros(100) + 1)):
        """Expanded dataSet class; contains enough information to save a result file.
        
        Init arguments:
        name -- name of expandedDataSet
        measured_data -- dataSet of measured data
        model_data -- dataSet of model curve
        circuit -- ecm_circuits.Circuit object
        
        Keyword arguments:
        parameters -- list of fit parameters
        
        Attributes:
        name -- name of expandedDataSet
        data -- dataSet of measured data
        model -- dataSet of model curve
        circuit -- ecm_circuits.Circuit object
        parameters -- list of fit parameters"""
        self.name = name
        self.data = measured_data
        self.model = model_data
        self.circuit = circuit
        self.parameters = parameters

#########################################
##KEEPING TRACK OF ALL THE OLD DATASETS##
#########################################

class HistoryManager():
    def __init__(self):
        """Keeps track of all saved datasets.
        
        Attributes:
        dataset_dict -- dict of {name: expandedDataSet, ...}
        visible -- list of names of expandedDataSets that should be shown in the plots
        
        Methods:
        add_dataset -- add a new expandedDataSet to dataset_dict
        toggle_dataset_visibility -- make a given expandedDataSet visisble or invisible"""
        self.dataset_dict = {}
        self.visible = []

    def add_dataset(self, edset):
        """Add an expandedDataSet to History.
        
        Arguments:
        self
        edset -- new expandedDataSet"""
        self.dataset_dict[edset.name] = edset

    def toggle_dataset_visibility(self, name):
        """Toggle the visibility of an expandedDataSet as a non-interactive dataset.
        
        Arguments:
        name -- name of expandedDataSet to be toggled (in)visible"""
        if name in self.visible:
            self.visible.remove(name)
        else:
            self.visible.append(name)

#####################################################################################
##SELECTING A DIFFERENT DATASET FOR PLOTTING WITH OR WITHOUT PARAMETER MANIPULATION##
#####################################################################################

class DataSetSelectorWindow(tk.Toplevel):
    def __init__(self, history_manager):
        """Window for selecting different expandedDataSets. DECiM core awaits the value of the chosen_dataset attribute.
        
        Init arguments:
        history_manager -- HistoryManager object
        
        Attributes:
        chosen_dataset -- tk.StringVar with the name of the chosen expandedDataSet
        rbuttons -- list of tk.Radiobuttons used to select the different expandedDataSets
        select_button -- button to close the window"""
        super().__init__()
        self.title("Dataset overview")
        self.geometry("400x600")

        self.chosen_dataset = tk.StringVar()

        self.rbuttons = []
        for h in history_manager.dataset_dict:
            self.rbuttons.append(tk.Radiobutton(self, text = history_manager.dataset_dict[h].name, variable = self.chosen_dataset, value = history_manager.dataset_dict[h].name))
            self.rbuttons[-1].pack(anchor = tk.W)

        self.rbuttons[0].select()

        self.select_button = tk.Button(self, text = "Select and close", command = self.destroy)
        self.select_button.pack(side = "top")
