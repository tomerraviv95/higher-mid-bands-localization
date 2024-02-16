from typing import Dict, List

import numpy as np
import scipy.signal
import torch

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
        # finally find the peaks in the spectrum
        if second_dim is not None:
            norm_values = norm_values.reshape(-1, second_dim)
        regions = self._find_peaks_in_spectrum(norm_values, self.thresh, second_dim)
        return np.array(list(regions.keys())), norm_values

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

    def _find_peaks_in_spectrum(self, norm_values: np.ndarray, thresh: float, second_dim: int) -> Dict[
        np.ndarray, List]:
        # treat the spectrum as 1d if the second dim is None
        if second_dim is None:
            indices, _ = scipy.signal.find_peaks(norm_values, height=thresh)
            return {index: index for index in indices}
        # treat the spectrum as 2d
        return self._get_peaks_regions(norm_values, thresh)

    def _get_peaks_regions(self, norm_values: np.ndarray, thresh: float) -> Dict[np.ndarray, List]:
        sorted_indices = np.unravel_index(np.argsort(norm_values, axis=None)[::-1], norm_values.shape)
        peaks_regions = {}
        for ind in zip(sorted_indices[0], sorted_indices[1]):
            if norm_values[ind[0], ind[1]] > thresh:
                create_new_peak = True
                for main_ind in peaks_regions.keys():
                    if abs(main_ind[0] - ind[0]) < 7 and abs(main_ind[1] - ind[1]) < 7:
                        peaks_regions[main_ind].append(ind)
                        create_new_peak = False
                        break
                if create_new_peak:
                    peaks_regions[ind] = [ind]
        filtered_peaks_regions = {peak: region for peak, region in peaks_regions.items() if len(region) > 20}
        return filtered_peaks_regions
