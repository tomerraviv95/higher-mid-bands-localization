import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALGS_DICT, ALG_TYPE
from python_code.utils.basis_functions import compute_angle_options
from python_code.utils.peaks_filtering import merge_two, filter_peaks


class AngleEstimator2D:
    """
    Angles Estimator for the AOA
    """

    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.aoa_res)  # dictionary of spatial frequencies
        self._angle_options = compute_angle_options(np.sin(self.angles_dict), zoa=1, values=np.arange(conf.Nr_x))
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum, _ = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                              n_elements=conf.Nr_x)
        estimator = Estimation(AOA=self.angles_dict[self._indices])
        return estimator


class AngleEstimator3D:
    """
    Angles Estimator for the AOA and the ZOA
    """

    def __init__(self):
        self.aoa_angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.aoa_res)  # dictionary of spatial frequencies
        self.zoa_angles_dict = np.linspace(0, np.pi / 2, conf.zoa_res)
        self.calc_angle_options()
        self._angle_options = self._angle_options.reshape(conf.Nr_y * conf.Nr_x, -1).T
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def calc_angle_options(self):
        aoa_vector_ys = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.sin(self.zoa_angles_dict), np.arange(conf.Nr_y)).T
        aoa_vector_xs = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.cos(self.zoa_angles_dict), np.arange(conf.Nr_x)).T
        self._angle_options = (aoa_vector_xs.T[:, :, None] @ aoa_vector_ys.T[:, None, :]).T

    def estimate(self, y: np.ndarray):
        indices, self._spectrum, L_hat = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                            n_elements=conf.Nr_y * conf.Nr_x)
        aoa_indices = indices // (conf.zoa_res)
        zoa_indices = indices % (conf.zoa_res)
        merged = np.array(merge_two(aoa_indices, zoa_indices))
        peaks = filter_peaks(merged, L_hat)
        self._aoa_indices = peaks[:, 0]
        self._zoa_indices = peaks[:, 1]
        estimation = Estimation(AOA=self.aoa_angles_dict[self._aoa_indices],
                                ZOA=self.zoa_angles_dict[self._zoa_indices])
        return estimation
