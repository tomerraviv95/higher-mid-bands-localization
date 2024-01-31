import numpy as np
import scipy.signal
from findpeaks import stats

from python_code import conf


class CaponBeamforming:
    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, one_dimensional=True):
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        inv_cov = np.linalg.inv(cov)
        inv_cov = inv_cov / np.linalg.norm(inv_cov)
        norm_values = np.linalg.norm((basis_vectors.conj() @ inv_cov) * basis_vectors, axis=1)
        norm_values = 1 / norm_values
        if one_dimensional:
            indices, _ = scipy.signal.find_peaks(norm_values, height=self.thresh)
            return indices, norm_values, len(indices)
        norm_values = norm_values.reshape(-1, int(conf.K / (conf.BW * conf.T_res)))
        res = stats.topology2d(norm_values)
        indices = np.where(res['Xdetect'] > 2 * self.thresh)
        return indices, norm_values, len(indices)
