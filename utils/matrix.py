import numpy as np


def construct_distant_penalty_matrix(patch_dimension, sigma=4, sqrt=True, double_sqrt=True):
    """Construct a matrix that penalizes distant pixels in a patch.

        Parameters
        ----------
        patch_dimension : int
            The dimension of the patch.
        sigma : float
            The standard deviation of the Gaussian kernel.

        Returns
        -------
        R : np.ndarray
            The constructed penalty matrix.

    """
    x = np.arange(patch_dimension)
    y = np.arange(patch_dimension)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    xv = xv.reshape(-1, 1)
    yv = yv.reshape(-1, 1)
    nom = (xv - xv.T) ** 2 + (yv - yv.T) ** 2
    if sqrt:
        nom = np.sqrt(nom) 
        if double_sqrt:
            nom = np.sqrt(nom)
    R = np.exp(-nom/ (2 * sigma ** 2))

    return R  # (patch_dimension ** 2, patch_dimension ** 2)