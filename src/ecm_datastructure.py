"""Part of DECiM. This file contains the primary datastructure type. Last modified 6 December 2023 by Henrik Rodenburg.

Classes:
dataSet -- standard class for holding impedance data"""

###########
##IMPORTS##
###########

import numpy as np

#######################
##DATASTRUCTURE CLASS##
#######################

class dataSet():
    def __init__(self, freq = [1, 10], real = [1, 10], imag = [1, 2]):
        """Standard class for holding impedance data.
        
        Init arguments:
        freq -- list or NumPy array of frequencies
        real -- list or NumPy array of real impedance components
        imag -- list or NumPy array of imaginary impedance components
        
        Attributes:
        freq -- frequency in Hz
        real -- real impedance component in Ohm
        imag -- imaginary impedance component in Ohm
        amplitude -- impedance amplitude in Ohm
        phase -- impedance phase in radians"""
        self.freq = np.array(freq) #Frequency in Hz
        self.real = np.array(real) #'Real' part of the impedance in Ohm
        self.imag = np.array(imag) #'Imaginary' part of the impedance in Ohm
        self.amplitude = np.sqrt(self.real**2 + self.imag**2)
        self.phase = np.arctan(self.imag/self.real)
