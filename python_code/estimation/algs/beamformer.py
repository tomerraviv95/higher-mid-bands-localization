import numpy as np
import torch

from python_code import DEVICE


class Beamformer:
    """
    The Beamformer
    """
    def run(self, y: np.ndarray, basis_vectors: np.ndarray, second_dim: int = None, use_gpu=False):
        """
        y is channel observations
        basis vectors are the set/sets of beamforming vectors in the dictionary
        if second_dim is False then do a 1 dimensional peak search
        if second_dim is True do a 2 dimensions peak search
        return a tuple of the [max_ind, spectrum]
        """
        # compute the spectrum values for each basis vector
        norm_values = self._compute_beamforming_spectrum(basis_vectors, use_gpu, y)
        # find the peaks in the spectrum
        if second_dim is not None:
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
        else:
            maximum_ind = np.argmax(norm_values)
        return np.array([maximum_ind]), norm_values, 0

    def _compute_beamforming_spectrum(self, basis_vectors: np.ndarray, use_gpu: bool, y: np.ndarray):
        # compute with cpu
        if not use_gpu:
            aoas = basis_vectors[0]
            toas = basis_vectors[1]
            left_matmul = np.einsum('ij,jmk->imk', aoas.conj(), y)
            right_matmul = np.einsum('ijk,jm->imk', left_matmul, toas.conj().T)
            norm_values = np.linalg.norm(right_matmul, axis=2)
        # do calculations on GPU - much faster for big matrices
        else:
            y = torch.tensor(y).to(DEVICE)
            aoas = basis_vectors[0]
            toas = basis_vectors[1]
            left_matmul = torch.einsum('ij,jmk->imk', aoas.conj(), y)
            right_matmul = torch.einsum('ijk,jm->imk', left_matmul, toas.conj().T)
            norm_values = torch.linalg.norm(right_matmul, axis=2)
            norm_values = norm_values.cpu().numpy()
        return norm_values
