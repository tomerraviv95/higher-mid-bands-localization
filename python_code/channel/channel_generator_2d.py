# Generate scatter points
from typing import List, Tuple

import numpy as np

from python_code import conf
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_time_options, compute_angle_options
from python_code.utils.constants import C, ChannelBWType, DATA_COEF, Channel, P_0
from python_code.utils.path_loss import compute_path_loss, calc_power


def compute_gt_channel_parameters(bs_loc: np.ndarray, ue_pos: np.ndarray, scatterers: np.ndarray, band: Band) -> Tuple[
    List[float], List[float], List[float]]:
    """"
    Computes the parameters for each path. Each path includes the toa, aoa and power.
    """
    # Initialize the channel parameters for L paths
    TOA = [0 for _ in range(conf.L)]
    AOA = [0 for _ in range(conf.L)]
    POWER = [0 for _ in range(conf.L)]
    # Compute for the LOS path
    TOA[0] = np.linalg.norm(ue_pos - bs_loc) / C
    AOA[0] = np.arctan2(ue_pos[1] - bs_loc[1], ue_pos[0] - bs_loc[0])
    initial_power = P_0 * np.sqrt(1 / 2) * (1 + 1j)
    POWER[0] = calc_power(initial_power, bs_loc, ue_pos, band.fc) / compute_path_loss(TOA[0], band.fc)
    # Compute for the NLOS paths
    for l in range(1, conf.L):
        AOA[l] = np.arctan2(scatterers[l - 1, 1] - bs_loc[1], scatterers[l - 1, 0] - bs_loc[0])
        TOA[l] = (np.linalg.norm(bs_loc - scatterers[l - 1]) + np.linalg.norm(conf.ue_pos - scatterers[l - 1])) / C
        initial_power = P_0 * np.sqrt(1 / 2) * (1 + 1j)
        POWER[l] = calc_power(calc_power(initial_power, bs_loc, scatterers[l - 1], band.fc), scatterers[l - 1],
                              ue_pos, band.fc) / compute_path_loss(TOA[l], band.fc)
    # assert that the toas are supported, must be smaller than largest distance divided by the speed
    assert all([TOA[l] < band.K / band.BW for l in range(len(TOA))])
    return TOA, AOA, POWER


def compute_observations(TOA: List[float], AOA: List[float], POWER: List[float], band: Band) -> np.ndarray:
    """"
    Compute the channel observations based on the band's parameters, and L TOAs, AOAs and POWERs
    """
    # For the covariance to have full rank we need to have enough samples, strictly more than the dimensions
    Ns = band.Nr_x * band.K * DATA_COEF
    # Initialize the observations and beamformers
    y = np.zeros((band.Nr_x, band.K, Ns), dtype=complex)
    # Generate multiple samples
    for ns in range(Ns):
        # Generate channel
        h = np.zeros((band.Nr_x, band.K), dtype=complex)
        # for each path
        for l in range(conf.L):
            # assume random phase beamforming
            F = np.exp(1j * np.random.rand(1) * 2 * np.pi)
            # phase delay for the K subcarriers
            delays_phase_vector = compute_time_options(band.fc, band.K, band.BW, np.array([TOA[l]]))
            if conf.channel_bandwidth == ChannelBWType.NARROWBAND.name:
                # different phase in each antennas element
                aoa_vector = compute_angle_options(np.sin(np.array([AOA[l]])), zoa=1, values=np.arange(band.Nr_x)).T
                delay_aoa_matrix = np.matmul(aoa_vector, delays_phase_vector)
            elif conf.channel_bandwidth == ChannelBWType.WIDEBAND.name:
                raise ValueError("Wideband is currently no supported!!")
            else:
                raise ValueError("No such type of channel BW!")
            # add for each path
            h += F * POWER[l] * delay_aoa_matrix
        # adding the white Gaussian noise
        noise = conf.sigma / np.sqrt(2) * (np.random.randn(band.Nr_x, band.K) + 1j * np.random.randn(band.Nr_x, band.K))
        # finally sum up to y, the final observation
        y[:, :, ns] = h + noise
    return y


def get_2d_channel(bs_loc: np.ndarray, ue_pos: np.ndarray, scatterers: np.ndarray, band: Band) -> Channel:
    # compute the parameters for each of the L paths
    TOA, AOA, POWER = compute_gt_channel_parameters(bs_loc, ue_pos, scatterers, band)
    # compute the channel observations based on the above paths
    y = compute_observations(TOA, AOA, POWER, band)
    # save results for easy access in a namedtuple
    channel_instance = Channel(scatterers=scatterers, y=y, TOA=TOA, AOA=AOA, band=band, ZOA=None)
    return channel_instance
