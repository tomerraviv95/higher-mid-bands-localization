from typing import List

import numpy as np
import torch.cuda

from python_code import conf, DEVICE
from python_code.channel.ny_channel.ny_channel_loader import load_ny_scenario
from python_code.channel.synthetic_channel.bs_scatterers import create_bs_locs, create_scatter_points
from python_code.channel.synthetic_channel.synthetic import generate_synthetic_parameters
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_angle_options, compute_time_options
from python_code.utils.constants import Channel, ScenarioType, SYNTHETIC_L_MAX, MEGA, NF, N_0, NS
from python_code.utils.path_loss import watt_power_from_dbm


def compute_observations(toas: List[float], aoas: List[float], powers: List[float], band: Band) -> np.ndarray:
    """"
    Compute the channel observations based on the band's parameters, and L toas, aoas and powers.
    See the paper for the definition of the channel.
    """
    # extract number of detectable paths
    L = len(powers)
    # Generate channel
    h = np.zeros((band.Nr, band.K, NS), dtype=complex)
    if torch.cuda.is_available():
        h = torch.tensor(h, dtype=torch.complex128).to(DEVICE)
    # calculate the noise power, and instead of multiplication by the noise, divide the signal
    BW_loss = 10 * np.log10(band.BW * MEGA)  # BW loss in dB
    noise_amp = watt_power_from_dbm(NF + BW_loss + N_0)
    # for each path
    for l in range(L):
        # assume random phase beamforming
        F = np.exp(1j * np.random.rand(1, 1, NS) * 2 * np.pi)
        # phase delay for the K subcarriers
        delays_phase_vector = compute_time_options(band.fc, band.K, band.BW, np.array([toas[l]]))
        # different phase in each antennas element
        aoa_vector = compute_angle_options(np.sin(np.array([aoas[l]])), values=np.arange(band.Nr)).T
        if torch.cuda.is_available():
            delay_aoa_tensor = torch.matmul(torch.tensor(aoa_vector).to(DEVICE),
                                            torch.tensor(delays_phase_vector).to(DEVICE))
            delay_aoa_tensor = torch.tensor(F).to(DEVICE) * delay_aoa_tensor.unsqueeze(-1)
            # add for each path
            h += watt_power_from_dbm(powers[l]) / noise_amp * delay_aoa_tensor
        else:
            delay_aoa_matrix = np.matmul(aoa_vector, delays_phase_vector)
            delay_aoa_matrix = F * delay_aoa_matrix[..., np.newaxis]
            # add for each path
            h += watt_power_from_dbm(powers[l]) / noise_amp * delay_aoa_matrix
    if torch.cuda.is_available():
        h = h.cpu().numpy()
    # adding the white Gaussian noise
    normal_gaussian_noise = 1 / np.sqrt(2) * (
            np.random.randn(band.Nr, band.K, NS) + 1j * np.random.randn(band.Nr, band.K, NS))
    # finally sum up to y, the final observation
    y = h + normal_gaussian_noise
    return y


def get_channel(bs_ind: int, ue_pos: np.ndarray, band: Band) -> Channel:
    if conf.scenario == ScenarioType.SYNTHETIC.name:
        bs_loc = create_bs_locs(bs_ind)
        scatterers = create_scatter_points(SYNTHETIC_L_MAX)
        toas, aoas, powers = generate_synthetic_parameters(bs_loc, ue_pos, scatterers, band)
    elif conf.scenario == ScenarioType.NY.name:
        bs_loc, toas, aoas, powers = load_ny_scenario(bs_ind, ue_pos, band)
    else:
        raise ValueError("Scenario is not implemented!!!")
    # compute the channel observations based on the above paths
    y = compute_observations(toas, aoas, powers, band)
    # save results for easy access in a namedtuple
    channel_instance = Channel(bs=bs_loc, y=y, TOA=toas, AOA=aoas, band=band, ZOA=None)
    return channel_instance
