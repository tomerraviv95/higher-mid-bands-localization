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

* dimensions - number of dimensions in the simulation. Either 'Two' or 'Three'.
* ue_pos - the 2d/3d position of the ue. For example, [10,10] in 2d.
* L - number of paths in the simulation, including LOS. You need to introduce additional scattering objects if you wish to have more than 5 paths.
* B - number of BSs by which localization is done.  You need to introduce additional ones in the bs script if you wish to have more than 3.
* K - number of subcarriers per sub-bands. A list with the number per sub-bands, e.g. [ 12 , 48 ] for two bands.
* Nr_x - number of antenna n_elements in the ULA for the 2d simulation, and the number of elements on the x plane URA for 3d simulation. Can handle a list of multiple sub-bands, e.g. [ 8 , 32 ]. 
* Nr_y - number of antennas on the y plane for the 3d simulation. For 2d - leave as 1. Can handle a list of multiple sub-bands, e.g. [ 8 , 32 ]. 
* sigma - the complex noise standard deviation value, e.g. a scalar 1.
* fc - list of carrier frequencies in MHz for each sub-band, e.g. [ 6000 , 24000 ].
* BW -  bandwidth of the sub-band in MHz, e.g. [ 100 , 400 ].
* channel_bandwidth - if the spatial bandwidth effect is present or not. Either 'NARROWBAND' or 'WIDEBAND'.

#### beamforming parameters
* est_type - what parameters should be estimated from the signal. 'time' for toa only, 'angle' for aoa (and zoa in 3d case), or 'both' for both angle and time.
* aoa_res - resolution in degrees for the azimuth dictionary, e.g. 1.5.
* zoa_res - resolution in degrees for the zenith dictionary, e.g. 2.
* T_res - resolution in mu seconds in the time dictionary.e.g. 0.002.

#### general
* seed - set random seed for the current run, e.g. 0. For the sake of reproducibility. 
* plot_estimation_results - boolean. Whether to plot the beamforming spectrum along with the peaks.

### detectors

The backbone detectors and their respective training: ViterbiNet, DeepSIC, Meta-ViterbiNet, Meta-DeepSIC, RNN BlackBox and FC BlackBox. The meta and non-meta detectors have slightly different API so they are seperated in the trainer class below. The meta variation has to receive the parameters of the original architecture for the training. The trainers are wrappers for the training and evaluation of the detectors. Trainer holds the training, sequential evaluation of pilot + info blocks. It also holds the main function 'eval' that trains the detector and evaluates it, returning a list of coded ber/ser per block. The train and test dataloaders are also initialized by the trainer. Each trainer is executable by running the 'evaluate.py' script after adjusting the config.yaml hyperparameters and choosing the desired method.

### plotters

Features main plotting tools for the paper:

* plotter_main - main plotting script used to get the figures in the paper. Based on the chosen PlotType enum loads the relevant config and runs the experiment.
* plotter_config - holds a mapping from the PlotType enum to the experiment's hyperparameters and setup.
* plotter_utils - colors, markers and linestyles for all evaluated methods, and the main plotting functions.
* plotter_methods - additional methods used for plotting.

### utils

Extra utils for pickle manipulations and tensor reshaping; calculating the accuracy over FER and BER; several constants; and the config singleton class.
The config works by the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar. 

The config is accessible from every module in the package, featuring the next parameters:
1. seed - random number generator seed. Integer.
2. channel_type - run either siso or mimo setup. Values in the set of ['SISO','MIMO']. String.
3. channel_model - chooses the channel taps values, either synthetic or based on COST2100. String in the set ['Cost2100','Synthetic'].
4. detector_type - selects the training + architecture to run. Short description of each option: 
* 'joint_black_box - Joint training of the black-box fully connected detector in the MIMO case.
* 'online_black_box' - Online training of the black-box fully connected detector in the MIMO case.
* 'joint_deepsic' - Joint training of the DeepSIC detector in the MIMO case.
* 'online_deepsic' - Online training of the DeepSIC detector in the MIMO case.
* 'meta_deepsic' - Online meta-training of the DeepSIC detector in the MIMO case.
* 'joint_rnn' - Joint training of the RNN detector in the SISO case.
* 'online_rnn' - online training of the RNN detector in the SISO case.
* 'joint_viterbinet' - Joint training of the ViterbiNet equalizer in the SISO case.
* 'online_viterbinet' - Online training of the ViterbiNet equalizer in the SISO case.
* 'meta_viterbinet' - Online meta-training of the ViterbiNet equalizer in the SISO case.
5. linear - whether to apply non-linear tanh at the channel output, not used in the paper but still may be applied. Bool.
6.fading_in_channel - whether to use fading. Relevant only to the synthetic channel. Boolean flag.
7. snr - signal-to-noise ratio, determines the variance properties of the noise, in dB. Float.
8. modulation_type - either 'BPSK' or 'QPSK', string.
9. memory_length - siso channel hyperparameter, integer.
10. n_user - mimo channel hyperparameter, number of transmitting devices. Integer.
11. n_ant - mimo channel hyperparameter, number of receiving devices. Integer.
12. block_length - number of coherence block bits, total size of pilot + data. Integer.
13. pilot_size - number of pilot bits. Integer.
14. blocks_num - number of blocks in the tranmission. Integer.
15. loss_type - 'CrossEntropy', could be altered to other types 'BCE' or 'MSE'.
16. optimizer_type - 'Adam', could be altered to other types 'RMSprop' or 'SGD'.
17. joint_block_length - joint training hyperparameter. Offline training block length. Integer.
18. joint_pilot_size - joint training hyperparameter. Offline training pilots block length. Integer.
19. joint_blocks_num - joint training hyperparameter. Number of blocks to train on offline. Integer.
20. joint_snrs - joint training hyperparameter. Number of SNRs to traing from offline. List of float values.
21. aug_type - what augmentations to use. leave empty list for no augmentations, or add whichever of the following you like: ['geometric_augmenter','translation_augmenter','rotation_augmenter']
22. online_repeats_n - if using augmentations, adds this factor times the number of pilots to the training batch. Leave at 0 if not using augmentations, if using augmentations try integer values in 2-5.

## resources

Keeps the COST channel coefficients vectors. Also holds config runs for the paper's numerical comparisons figures.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 3060 with CUDA 12. 

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

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\environment\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
