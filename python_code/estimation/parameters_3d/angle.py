from typing import List, Union

import numpy as np

from python_code import conf
from python_code.estimation.algs import ALGS_DICT, ALG_TYPE
from python_code.estimation.parameters_2d.angle import AngleEstimator2D
from python_code.estimation.parameters_3d import coef_per_frequencies_dict
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_angle_options
from python_code.utils.constants import Estimation, BandType


class AngleEstimator3D(AngleEstimator2D):
    """
    Angles Estimator for the AOA and the ZOA
    """

    def __init__(self, bands: List[Band]):
        self.zoa_angles_dict = np.arange(0, np.pi / 2, conf.zoa_res * np.pi / 180)
        super(AngleEstimator3D, self).__init__(bands)

    def calc_angle_options(self, Nr_x: int, Nr_y: int):
        aoa_vector_ys = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.sin(self.zoa_angles_dict), np.arange(Nr_y)).T
        aoa_vector_xs = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.cos(self.zoa_angles_dict), np.arange(Nr_x)).T
        mat = (aoa_vector_xs.T[:, :, None] @ aoa_vector_ys.T[:, None, :]).T
        return mat.reshape(Nr_y * Nr_x, -1).T

    def _single_band_constructor(self, bands: List[Band]):
        band = bands[0]
        self.n_elements = band.Nr_y * band.Nr_x
        self._angle_options = self.calc_angle_options(band.Nr_x, band.Nr_y)
        self.algorithm = ALGS_DICT[ALG_TYPE][BandType.SINGLE](coef_per_frequencies_dict[band.fc])

    def _multiband_constructor(self, bands: List[Band]):
        self.n_elements = [band.Nr_y * band.Nr_x for band in bands]
        self._angle_options = [self.calc_angle_options(band.Nr_x, band.Nr_y) for band in bands]
        self.algorithm = ALGS_DICT[ALG_TYPE][BandType.MULTI](coef_per_frequencies_dict[bands[-1].fc])

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        self._indices, self._spectrum, _ = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                              n_elements=self.n_elements,
                                                              second_dim=len(self.zoa_angles_dict))
        # if no peaks found - return an empty estimation
        if len(self._indices) == 0:
            return Estimation()
        self._aoa_indices = self._indices[:, 0]
        self._zoa_indices = self._indices[:, 1]
        estimation = Estimation(AOA=self.aoa_angles_dict[self._aoa_indices],
                                ZOA=self.zoa_angles_dict[self._zoa_indices])
        return estimation
