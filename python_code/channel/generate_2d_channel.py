from typing import List

import numpy as np

from python_code import conf
from python_code.channel import create_bs_locs_2d, create_scatter_points_2d
from python_code.channel.ny_channel.ny_channel_loader_2d import load_ny_scenario
from python_code.channel.synthetic_channel.synthetic_2d import generate_synthetic_parameters
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_angle_options, compute_time_options
from python_code.utils.constants import Channel, ChannelBWType, DATA_COEF, ScenarioType


def compute_observations(TOA: List[float], AOA: List[float], POWER: List[float], band: Band) -> np.ndarray:
    """"
    Compute the channel observations based on the band's parameters_2d, and L TOAs, AOAs and POWERs
    """
    L = len(POWER)
    # For the covariance to have full rank we need to have enough samples, strictly more than the dimensions
    Ns = int(band.Nr_x * band.K * DATA_COEF)
    # Initialize the observations and beamformers
    y = np.zeros((band.Nr_x, band.K, Ns), dtype=complex)
    # Generate multiple samples
    for ns in range(Ns):
        # Generate channel
        h = np.zeros((band.Nr_x, band.K), dtype=complex)
        # for each path
        for l in range(L):
            # assume random phase beamforming
            F = np.exp(1j * np.random.rand(1) * 2 * np.pi)
            # phase delay for the K subcarriers
            delays_phase_vector = compute_time_options(band.fc, band.K, band.BW, np.array([TOA[l]]))
            if conf.channel_bandwidth == ChannelBWType.NARROWBAND.name:
                # different phase in each antennas element
                aoa_vector = compute_angle_options(np.sin(np.array([AOA[l]])), zoa=1, values=np.arange(band.Nr_x)).T
                delay_aoa_matrix = np.matmul(aoa_vector, delays_phase_vector)
            elif conf.channel_bandwidth == ChannelBWType.WIDEBAND.name:
                raise ValueError("Wideband is currently not supported!!")
            else:
                raise ValueError("No such type of channel BW!")
            # add for each path
            h += F * POWER[l] * delay_aoa_matrix
        # adding the white Gaussian noise
        noise = conf.sigma / np.sqrt(2) * (np.random.randn(band.Nr_x, band.K) + 1j * np.random.randn(band.Nr_x, band.K))
        # finally sum up to y, the final observation
        y[:, :, ns] = h + noise
    return y


def get_2d_channel(bs_ind: int, ue_pos: np.ndarray, band: Band) -> Channel:
    if conf.scenario == ScenarioType.SYNTHETIC.name:
        bs_loc = create_bs_locs_2d(bs_ind)
        scatterers = create_scatter_points_2d(conf.L)
        TOA, AOA, POWER = generate_synthetic_parameters(bs_loc, ue_pos, scatterers, band)
    elif conf.scenario == ScenarioType.NY.name:
        bs_loc, TOA, AOA, POWER = load_ny_scenario(bs_ind, ue_pos, band)
        scatterers = None
    else:
        raise ValueError("Scenario is not implemented!!!")
    # compute the channel observations based on the above paths
    y = compute_observations(TOA, AOA, POWER, band)
    # save results for easy access in a namedtuple
    channel_instance = Channel(scatterers=scatterers, bs=bs_loc, y=y, TOA=TOA, AOA=AOA, band=band, ZOA=None)
    return channel_instance
