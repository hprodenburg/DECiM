# DEC*i*M

DEC*i*M (Determination of Equivalent Circuit Models) is an equivalent circuit model fitting program for impedance data. It is a GUI-based program, written entirely in Python.

## Prerequisites

The current DEC*i*M version (1.3.2) requires the following software to already be installed on your computer:

- Python 3.10 or higher (including copy, webbrowser, functools and tkinter as part of the Python Standard Library)
- NumPy
- SciPy
- Matplotlib
- Optax

**WARNING: DEC*i*M has been reported not to work well together with Conda. Please do not use a Python 3 installation with Conda to install and run DEC*i*M.**

## Installation

If the prerequisites are not yet installed, install them. Begin with Python 3.10+ (it may be convenient to choose WinPython if you are using Windows, since this is the same distribution on which DEC*i*M was developed and you can expect few or no surprises). Then install the libraries.

Once everything is ready, download the entire DEC*i*M project. DEC*i*M can be started immediately.

For detailed instructions, please refer to the *Installation_guide_from_scratch.md* file.

## Usage

Open your preferred program to access the command line and navigate to the `src` directory of the DEC*i*M distribution. Provided your Python installation allows you to run Python 3 as `python`, you can start DEC*i*M from the command line as follows:

```
python DECiM.py
```

For detailed operating instructions, please consult the PDF manual.

## Citing

If you use DEC*i*M for work leading to a scholarly publication, please cite the publication describing DEC*i*M:

Rodenburg, H.P. and Ngene, P. DEC*i*M: Determination of equivalent circuit models. *SoftwareX* **2024**, *27*, 101807. https://doi.org/10.1016/j.softx.2024.101807