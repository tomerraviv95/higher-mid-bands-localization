from typing import Tuple

import numpy as np
import scipy.signal
import torch
from scipy.ndimage.measurements import label

from python_code import DEVICE


class CaponBeamforming:
    """
    The Capon Beamformer.
    Lorenz, R.G. and Boyd, S.P., 2005. Robust minimum variance beamforming.
    IEEE transactions on signal processing, 53(5), pp.1684-1696.
    """

    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, second_dim: int = None,
            third_dim: int = None, use_gpu=False):
        """
        y is channel observations
        basis vectors are the set of beamforming vectors in the dictionary
        n_elements is the dimension of the covariance matrix, .e.g antennas num for AOA, subcarriers num for TOA
        if second_dim is None then do a 1 dimensional peak search
        if second_dim is not None but third_dim is None do a 2 dimensions peak search
        if both are not None, do a 3 dimensions peak search
        batches allow for batching the computations of the beamforming objective to avoid memory crash
        return a tuple of the [max_indices, spectrum, L_hat] where L_hat is the estimated number of paths
        """
        # compute inverse covariance matrix
        cov = self._compute_cov(n_elements, y, use_gpu)
        # compute the Capon spectrum values for each basis vector
        norm_values = self._compute_capon_spectrum(basis_vectors, use_gpu, cov)
        # finally find the peaks in the spectrum
        return self.find_peaks_in_spectrum(norm_values, second_dim, third_dim)

    def _compute_cov(self, n_elements: int, y: np.ndarray, use_gpu: bool):
        if not use_gpu:
            cov = np.cov(y.reshape(n_elements, -1))
            cov = np.linalg.inv(cov)
            cov = cov / np.linalg.norm(cov)
        else:
            y = torch.tensor(y, dtype=torch.cfloat).to(DEVICE)
            cov = torch.cov(y.reshape(n_elements, -1))
            cov = torch.inverse(cov)
            cov = cov / torch.norm(cov)
        return cov

    def _compute_capon_spectrum(self, basis_vectors: np.ndarray, use_gpu: bool, cov: np.ndarray):
        # compute with cpu - no cpu/memory issues
        if not use_gpu:
            norm_values = np.linalg.norm((basis_vectors.conj() @ cov) * basis_vectors, axis=1)
            norm_values = 1 / norm_values
        # do calculations on GPU - much faster for big matrices
        else:
            cov_mat = torch.tensor(cov, dtype=torch.cfloat).to(DEVICE)
            res0, max_batches = basis_vectors(batch_ind=0)
            batch_size = res0.shape[0]
            norm_values = torch.zeros(batch_size * max_batches, dtype=torch.float).to(DEVICE)
            for i in range(max_batches):
                cur_basis_vectors = basis_vectors(batch_ind=i)[0]
                norm_values[i * batch_size:(i + 1) * batch_size] = torch.linalg.norm(
                    torch.matmul(cur_basis_vectors.conj(), cov_mat)
                    * cur_basis_vectors, dim=1)
            norm_values = (1 / norm_values).cpu().numpy().astype(float)
        return norm_values

    @staticmethod
    def get_current_component(norm_values: np.ndarray, component_indx: np.ndarray, third_dim: int):
        if third_dim is None:
            if component_indx[0] < norm_values.shape[0] and component_indx[1] < norm_values.shape[1]:
                return norm_values[component_indx[0]][component_indx[1]]
            return None
        return norm_values[component_indx[0]][component_indx[1]][component_indx[2]]

    def find_peaks_in_spectrum(self, norm_values: np.ndarray, second_dim: int, third_dim: int) -> Tuple[
        np.ndarray, np.ndarray, int]:
        # treat the spectrum as 1d if the second dim is None
        if second_dim is None:
            indices, _ = scipy.signal.find_peaks(norm_values, height=self.thresh)
            return indices, norm_values, len(indices)
        # treat the spectrum as 2d if the third dim is None
        elif third_dim is None:
            norm_values = norm_values.reshape(-1, second_dim)
            labeled, ncomponents = label(norm_values > self.thresh, structure=np.ones((3, 3), dtype=int))
        # treat the spectrum as 3d
        else:
            norm_values = norm_values.reshape(-1, second_dim, third_dim)
            labeled, ncomponents = label(norm_values > self.thresh,
                                         structure=np.ones((3, 3, 3), dtype=int))

        # in case of 2d or 3d spectrum, finding regions of peaks then taking the maximum in each region
        # as the representative for that region
        indices = []
        for component in range(1, ncomponents + 1):
            # get the region indices
            component_indices = np.array(np.where(labeled == component)).T
            max, ind = 0, None
            # look for the maximum value and indices in that region
            for component_indx in component_indices:
                cur_comp = self.get_current_component(norm_values, component_indx, third_dim)
                if cur_comp > max:
                    max = cur_comp
                    ind = component_indx
            # add the corresponding index of the maximum value
            indices.append(ind)
        return np.array(indices), norm_values, len(indices)
