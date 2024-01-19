import numpy as np

from python_code import conf
from python_code.estimation import ALG_TYPE, ALGS_DICT, Estimation
from python_code.utils.basis_functions import create_wideband_aoa_mat, compute_angle_options


class AngleEstimator2D:
    """
    Angles Estimator for the AOA
    """

    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.aoa_res)  # dictionary of spatial frequencies
        self._angle_options = compute_angle_options(self.angles_dict, zoa=1, values=np.arange(conf.Nr_x))
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
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
        ## the bottom code is the vectorized implementation of the easier to understand code below
        ## runs over each aoa, and for it on each zoa and calculates the steering vector option
        # self._angle_options = np.zeros((conf.Nr_y, conf.Nr_x, conf.zoa_res * conf.aoa_res), dtype=complex)
        # for i in range(conf.aoa_res):
        #     aoa = self.aoa_angles_dict[i]
        #     for j in range(conf.zoa_res):
        #         zoa = self.zoa_angles_dict[j]
        #         aoa_vector_y = compute_angle_options(np.array([aoa]).reshape(-1, 1),
        #                                              np.sin(np.array([zoa])), np.arange(conf.Nr_y)).T
        #         aoa_vector_x = compute_angle_options(np.array([aoa]).reshape(-1, 1),
        #                                              np.cos(np.array([zoa])), np.arange(conf.Nr_x)).T
        #         self._angle_options[:, :, i * conf.zoa_res + j] = aoa_vector_y @ aoa_vector_x.T
        aoa_vector_ys = compute_angle_options(self.aoa_angles_dict.reshape(-1, 1),
                                              np.sin(self.zoa_angles_dict), np.arange(conf.Nr_y)).T
        aoa_vector_xs = compute_angle_options(self.aoa_angles_dict.reshape(-1, 1),
                                              np.cos(self.zoa_angles_dict), np.arange(conf.Nr_x)).T
        self._angle_options = (aoa_vector_xs.T[:, :, None] @ aoa_vector_ys.T[:, None, :]).T

    def estimate(self, y: np.ndarray):
        indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                     n_elements=conf.Nr_y * conf.Nr_x)
        self._aoa_indices = indices // (conf.zoa_res)
        self._zoa_indices = indices % (conf.zoa_res)
        estimation = Estimation(AOA=self.aoa_angles_dict[self._aoa_indices],
                                ZOA=self.zoa_angles_dict[self._zoa_indices])
        return estimation


class WidebandAngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.aoa_res)  # dictionary of spatial frequencies
        self._angle_options = create_wideband_aoa_mat(self.angles_dict, conf.K, conf.BW, conf.fc, conf.Nr,
                                                      stack_axis=1).reshape(conf.Nr * conf.K, -1).T
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                           n_elements=conf.Nr * conf.K)
        estimator = Estimation(AOA=self.angles_dict[self._indices])
        return estimator
