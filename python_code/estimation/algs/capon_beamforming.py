import numpy as np
import torch
from scipy.ndimage import label

from python_code import DEVICE
from python_code.utils.constants import MAX_COMPONENTS


class CaponBeamforming:
    """
    The Capon Beamformer.
    Lorenz, R.G. and Boyd, S.P., 2005. Robust minimum variance beamforming.
    IEEE transactions on signal processing, 53(5), pp.1684-1696.
    """

    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, second_dim: int = None,
            use_gpu=False):
        """
        y is channel observations
        basis vectors are the set of beamforming vectors in the dictionary
        n_elements is the dimension of the covariance matrix, .e.g antennas num for AOA, subcarriers num for TOA
        if second_dim is None then do a 1 dimensional peak search
        if second_dim is not None do a 2 dimensions peak search
        batches allow for batching the computations of the beamforming objective to avoid memory crash
        return a tuple of the [max_indices, spectrum, L_hat] where L_hat is the estimated number of paths
        """
        # compute inverse covariance matrix
        cov = self._compute_cov(n_elements, y, use_gpu)
        # compute the Capon spectrum values for each basis vector
        norm_values = self._compute_capon_spectrum(basis_vectors, use_gpu, cov)
        # find the peaks in the spectrum
        if second_dim is not None:
            norm_values = norm_values.reshape(-1, second_dim)
            maximum_ind = np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape)
            return np.array([maximum_ind]), norm_values, 0
            #
            # labeled, ncomponents = self.label_spectrum_by_peaks(norm_values)
            # if ncomponents == 0:
            # s_groups = self.compute_peaks_groups(labeled, ncomponents, norm_values)
            # # minimal TOA, maximum power peak
            # maximum_ind = s_groups[0][2]
            # return np.array([maximum_ind]), norm_values, 0
        maximum_ind = np.argmax(norm_values)
        return np.array([maximum_ind]), norm_values, 0

    def label_spectrum_by_peaks(self, norm_values):
        max_indices = np.argsort(norm_values, axis=None)[::-1][:MAX_COMPONENTS]
        indices = np.array(np.unravel_index(max_indices, norm_values.shape)).T
        max_val = norm_values[indices[0][0], indices[0][1]]
        image = np.zeros_like(norm_values)
        for ind in indices:
            if norm_values[ind[0], ind[1]] > max(0.3 * max_val, self.thresh):
                image[ind[0], ind[1]] = 1
        labeled, ncomponents = label(image, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
        return labeled, ncomponents

    def compute_peaks_groups(self, labeled, ncomponents, norm_values):
        groups = []
        for comp_ind in range(1, 1 + ncomponents):
            group_inds = np.array(np.where(labeled == comp_ind)).T
            group_toa = group_inds.min(axis=0)[1]
            group_max_power = max(norm_values[group_inds[:, 0], group_inds[:, 1]])
            peak_ind = group_inds[np.argmax(norm_values[group_inds[:, 0], group_inds[:, 1]])]
            groups.append((group_toa, group_max_power, peak_ind))
        s_groups = sorted(groups, key=lambda x: (x[0], -x[1]))
        s_groups = list(filter(lambda x: x[0] == s_groups[0][0], s_groups))
        return s_groups

    def _compute_cov(self, n_elements: int, y: np.ndarray, use_gpu: bool):
        cov = np.cov(y.reshape(n_elements, -1))
        if not use_gpu:
            cov = np.linalg.inv(cov)
            cov = cov / np.linalg.norm(cov)
        else:
            cov = torch.tensor(cov).to(DEVICE)
            cov = torch.linalg.inv(cov)
            cov = cov / torch.linalg.norm(cov)
            cov = cov.type(torch.cfloat)
        return cov

    def _compute_capon_spectrum(self, basis_vectors: np.ndarray, use_gpu: bool, cov: np.ndarray):
        # compute with cpu - no cpu/memory issues
        if not use_gpu:
            norm_values = np.linalg.norm((basis_vectors.conj() @ cov) * basis_vectors, axis=1)
            norm_values = 1 / norm_values
        # do calculations on GPU - much faster for big matrices
        else:
            res0, max_batches = basis_vectors(batch_ind=0)
            batch_size = res0.shape[0]
            norm_values = torch.zeros(batch_size * max_batches).to(DEVICE)
            for i in range(max_batches):
                cur_basis_vectors = basis_vectors(batch_ind=i)[0].type(torch.cfloat)
                multiplication = torch.matmul(cur_basis_vectors.conj(), cov)
                norm_values[i * batch_size:(i + 1) * batch_size] = torch.linalg.norm(multiplication * cur_basis_vectors,
                                                                                     dim=1)
            norm_values = (1 / norm_values).cpu().numpy()
        return norm_values
