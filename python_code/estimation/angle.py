import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALGS_DICT, ALG_TYPE
from python_code.utils.basis_functions import compute_angle_options


class AngleEstimator2D:
    """
    Angles Estimator for the AOA
    """

    def __init__(self):
        self.aoa_angles_dict = np.arange(-np.pi / 2, np.pi / 2, conf.aoa_res * np.pi / 180)
        self._angle_options = compute_angle_options(np.sin(self.aoa_angles_dict), zoa=1, values=np.arange(conf.Nr_x))
        self.algorithm = ALGS_DICT[ALG_TYPE](5)

    def estimate(self, y):
        self._indices, self._spectrum, _ = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                              n_elements=conf.Nr_x)
        estimator = Estimation(AOA=self.aoa_angles_dict[self._indices])
        return estimator


class AngleEstimator3D(AngleEstimator2D):
    """
    Angles Estimator for the AOA and the ZOA
    """

    def __init__(self):
        super(AngleEstimator3D, self).__init__()
        self.zoa_angles_dict = np.arange(0, np.pi / 2, conf.zoa_res * np.pi / 180)
        self.calc_angle_options()
        self._angle_options = self._angle_options.reshape(conf.Nr_y * conf.Nr_x, -1).T
        self.algorithm = ALGS_DICT[ALG_TYPE](10)

    def calc_angle_options(self):
        aoa_vector_ys = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.sin(self.zoa_angles_dict), np.arange(conf.Nr_y)).T
        aoa_vector_xs = compute_angle_options(np.sin(self.aoa_angles_dict).reshape(-1, 1),
                                              np.cos(self.zoa_angles_dict), np.arange(conf.Nr_x)).T
        self._angle_options = (aoa_vector_xs.T[:, :, None] @ aoa_vector_ys.T[:, None, :]).T

    def estimate(self, y: np.ndarray):
        indices, self._spectrum, L_hat = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                            n_elements=conf.Nr_y * conf.Nr_x,
                                                            second_dim=len(self.zoa_angles_dict))
        self._aoa_indices = indices[:, 0]
        self._zoa_indices = indices[:, 1]
        estimation = Estimation(AOA=self.aoa_angles_dict[self._aoa_indices],
                                ZOA=self.zoa_angles_dict[self._zoa_indices])
        return estimation
