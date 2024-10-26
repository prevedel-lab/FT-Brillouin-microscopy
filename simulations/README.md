# Simulation for FT-Brillouin microscopy
This folder contains the script ["FTBM_precision_simulation.py"](FTBM_precision_simulation.py) to generate the simulation plots in [our preprint](https://doi.org/10.48550/arXiv.2409.02092). The details on how the simulation is performed can be found in the main text and "Methods" section.

Additionaly it contains the file ["FTBM_sign_flip_simulation.py"](FTBM_sign_flip_simulation.py) that simulates the effect on the spectrum of a sign flip in the interferogram (to compare it to the calculation in the Supplementary Information).

## Requirements
The script requires **numpy** and **scipy** for the computation and **plotly** to generate the plots. **tqdm** is also imported to show the progression of the simulation but it is not strictly required for the simulation.

The script was tested with Python 3.11.7, numpy 1.26.4, scipy 1.11.4 and plotly 5.9.0

## Run
You can download the script ["FTBM_precision_simulation.py"](FTBM_precision_simulation.py), install the required packages and run it.
The default renderer for plotly is set as 'svg', so you should expect to see the generated plots in your IDE; more information can be found [here](https://plotly.com/python/renderers/).
The plots are also saved as 'svg' files in the current folder.
