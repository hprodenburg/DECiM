# DEC*i*M installation from scratch: a complete guide

This guide covers all the steps for downloading and installing DEC*i*M on Windows.

## Python 3

Since DEC*i*M is a Python 3 package, running it requires that you have Python 3 installed. Specifically, you need Python 3.10 or higher (e.g., Python 3.12). DEC*i*M does not generally play nice with Conda, so the recommended installation route for Python is either getting it from the official website (python.org/downloads/) or by downloading and installing WinPython (winpython.github.io).

It is helpful to add Python 3 to PATH; the README and PDF Manual for DEC*i*M both assume that this is done. You can add Python 3 to PATH during the installation (see docs.python.org/3/using/windows.html) or afterwards **if you know where you installed Python** by opening Windows Explorer (this is the file system explorer) and going through the following steps:

1. Right-click ```This PC```.
2. Click ```Properties```.
3. Click ```Advanced system settings```.
4. Click ```Environment variables```
5. In the ```System variables``` box, add the directory in which *python.exe* is located to the variable ```Path```. Click ```Path```, and in the new window, click ```New```. Then type the name of Python 3 directory (e.g., *C:\Python\WPy64-31180\python-3.11.8.amd64*) into the new line.
6. Click ```OK``` to close the path variable editing window.
7. Click ```OK``` to close the system variables window.
8. Click ```Apply``` and then ```OK``` to close the properties window.

Now, you should check if you can start Python 3 by opening a terminal (```cmd```) and typing ```python```. Python 3 should start when you do this. You can confirm it works by typing ```print('test')```; the output should be ```test```. To close Python, type ```exit()``` and then ```exit``` to close the terminal.

## Dependencies

If you followed the installation instructions above, you have a working Python 3 installation that includes Pip. Pip is required to install the packages that DEC*i*M needs, but which are not included in the standard Python 3 installation. These packages are:

- NumPy (numpy.org)
- SciPy (scipy.org)
- Matplotlib (matplotlib.org)
- Optax (optax.readthedocs.io/en/latest/)

If you installed WinPython, you only need to install Optax. Otherwise, you need to install all four packages. To do this, open a terminal (```cmd```) and type in the commands specified (without quotation marks/backticks) in the table below.

| Package    | Installation command                      |
| :--------- | :---------------------------------------: |
| NumPy      | ```python -m pip install -U numpy```      |
| SciPy      | ```python -m pip install -U scipy```      |
| Matplotlib | ```python -m pip install -U matplotlib``` |
| Optax      | ```python -m pip install -U optax```      |

## DEC*i*M

To install DEC*i*M itself, either download or clone the GitHub repository (github.com/hprodenburg/DECiM). Downloading DEC*i*M is done by clicking ```Code``` and then ```Download ZIP```. Extract the .zip file to some directory you can remember.

To test the installation, open ```cmd```, navigate to the installation directory (where the *README.md* file is located) with the ```cd``` command (learn.microsoft.com/en-us/windows-server/administration/windows-commands/cd). Then, go to the ```src``` folder with

```cd src```

and start DEC*i*M with

```python DECiM.py```

DEC*i*M should now launch; give it a few seconds to load.