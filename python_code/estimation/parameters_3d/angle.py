import numpy as np

from python_code import conf
from python_code.estimation.algs import ALGS_DICT, ALG_TYPE
from python_code.estimation.parameters_2d.angle import AngleEstimator2D
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_angle_options
from python_code.utils.constants import Estimation


class AngleEstimator3D(AngleEstimator2D):
    """
    Angles Estimator for the AOA and the ZOA
    """

    def __init__(self, band: Band):
        super(AngleEstimator3D, self).__init__(band)
        self.Nr_y = band.Nr_y
        self.zoa_angles_dict = np.arange(0, np.pi / 2, conf.zoa_res * np.pi / 180)
        self.calc_angle_options()
        self._angle_options = self._angle_options.reshape(band.Nr_y * band.Nr_x, -1).T
        self.algorithm = ALGS_DICT[ALG_TYPE](10)

    def calc_angle_options(self):
        aoa_vector_ys = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.sin(self.zoa_angles_dict), np.arange(self.Nr_y)).T
        aoa_vector_xs = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.cos(self.zoa_angles_dict), np.arange(self.Nr_x)).T
        self._angle_options = (aoa_vector_xs.T[:, :, None] @ aoa_vector_ys.T[:, None, :]).T

    def estimate(self, y: np.ndarray) -> Estimation:
        self._indices, self._spectrum, _ = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                              n_elements=self.Nr_y * self.Nr_x,
                                                              second_dim=len(self.zoa_angles_dict))
        self._aoa_indices = self._indices[:, 0]
        self._zoa_indices = self._indices[:, 1]
        estimation = Estimation(AOA=self.aoa_angles_dict[self._aoa_indices],
                                ZOA=self.zoa_angles_dict[self._zoa_indices])
        return estimation
