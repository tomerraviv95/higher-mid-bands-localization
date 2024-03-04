*"Strength lies in differences, not in similarities."* 

--Stephen R. Covey.

# Multi-Frequency Localization in the FR3 band

Python repository for the paper "Multi-Frequency Localization in the FR3 band".

Please cite our [paper](https://arxiv.org/abs/2305.07309), if the code is used for publishing research. 

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [main](#main)
    + [config](#config)
    + [channel](#channel)
    + [estimation](#estimation)
    + [optimization](#optimization)
    + [plotting](#plotting)
    + [utils](#utils)
  * [resources](#resources)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This work aims to bridge the gap for multi-frequency localization in the FR3 band. We suggest a multi-frequency localization algorithm that combines the beamforming spectrum across different sub-bands in FR3, compensating for errors in different frequencies by employing multiple ones, which results in a more robust estimation of the TOA and AOA parameters. This, in turn, results in a more robust localization of the UE, as compared to estimation and localization using a single frequency sub-band.

# Folders Structure

## python_code 

The python simulations of the simulated setup, including: channel generation, parameters estimation and least squares optimization to locate the UE.

### main

This is the main script. It runs the generation of the channel observations at each base station as in the equations in the paper. Then, it performs the parameters estimations, including the TOA, AOA and ZOA parameters. Finally, if enough parameters are available, it can localize the user by solving the least squares problem.

### config 

You can control the simulated setup via the hyperparameters in the config:

#### System parameters
* scenario - either the new york scenario from the paper (set to 'NY') or a synthetic one (set as 'SYNTHETIC').
* ue_pos - the 2d position of the ue. For example, [10,10].
* B - number of BSs by which localization is done.  It is 1 for the NY scenario and maximum of 3 for the synthetic.
* input_power - input power in dBm.

#### Bands parameters
* K - number of subcarriers per sub-band. A list with the number per sub-bands, e.g. [ 12 , 48 ] for two bands.
* Nr - number of antenna elements in the ULA. A list of either single or multiple sub-bands, e.g. [ 8 , 32 ]. 
* fc - list of carrier frequencies in MHz for each sub-band, e.g. [ 6000 , 24000 ].
* BW -  bandwidth for each sub-band in MHz, e.g. [ 100 , 400 ].

#### beamforming parameters
* est_type - what parameters should be estimated from the signal. 'time' for toa only, 'angle' for aoa, or 'both' for both angle and time. UE localization is done only with 'both'.
* alg - the parameters estimation algorithm. Either 'Beamformer' or 'MUSIC'.
* aoa_res - resolution in degrees for the azimuth dictionary, e.g. 1.5.
* T_res - resolution in mu seconds in the time dictionary.e.g. 0.002.

#### general
* seed - set random seed for the current run, e.g. 1. For the sake of reproducibility. 
* plot_estimation_results - boolean. Whether to plot the beamforming spectrum along with the peaks.

### channel

Contains the scripts to generate the channel observations Y. 
For each UE and BS pair, we can calculate per link the parameters for each 
of the paths, 
including its channel power, toa and aoa for each path. 
Then, Y can be calculated as in the paper.

### estimation

Contains the time / angle / time and angle estimators. 
In each one we simply create the relevant beamforming basis vectors 
(a.k.a the dictionary) and call the relevant estimation algorithm. 
The algorithms supported are either the beamformer from the paper or MUSIC. 
Note that MUSIC is implemented only for the 2d multi-frequency case. 
The main script is estimate_physical_parameters.py that 
estimates the parameters for each BS based on the multiple available sub-bands.

### optimization

Contains the optimization script. Note that we neglect the NLOS paths, 
assuming LOS conditions exist. Thus, aoa and toa yield the direction and distance
of the UE as relative to each BS.

### plotting

Features main plotting tools for the paper. 
Also, holds the spectrum plotting scripts for each time/angle/both estimators. 

### utils

Holds additional utils that do not belong to only a single module from the above: 

* bands_manipulation - aggregate all the sub-bands hyperparameters, such as its BW, carrier frequency and number of antennas, into a single placeholder.
* basis_functions - create the phase vector for either TOA or the AOA. It is used both for the channel generation and for the beamforming estimators.
* config_singleton - the config singleton class is defined [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern).
* constants - some global constants such as the speed of light, and several Enums used across the simulation.
* path_loss - defining some concrete buildings (walls) in the simulation to create a sense of an urban environment. 

## resources

Keeps several configs for example and the ray-tracing data for each sub-band.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the main or one of the plotters.

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f environment.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\environment\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
