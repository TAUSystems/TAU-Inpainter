""" Contains algorithms for imputing missing regions of an image, such as 
    those caused by saturation, blocks, or calibration references.
    
    created by Reinier van Mourik, TAU Systems Inc <reinier.vanmourik@tausystems.com>
"""
from __future__ import annotations

from typing import Optional

from warnings import warn

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import vstack as sp_vstack
from scipy.sparse import eye as sp_eye
from scipy.sparse.linalg import spsolve as sp_solve

class TotalCurvatureInpainter:
    """ An inpainter that minimizes the total (squared) curvature of the 
        inpainted surface. 

        Algorithm designed and derived independently by Reinier van Mourik,
        but seems similar to 
            Chan, Tony F., and Jianhong (Jackie) Shen. Image Processing and Analysis: 
            Variational, PDE, Wavelet, and Stochastic Methods. Society for Industrial 
            and Applied Mathematics, 2005. https://doi.org/10.1137/1.9780898717877.
        section 6.8, or
            Shen, Jianhong, Sung Ha Kang, and Tony F. Chan. “Euler's Elastica and 
            Curvature-Based Inpainting.” SIAM Journal on Applied Mathematics 63, no. 2 
            (January 2003): 564-92. https://doi.org/10.1137/S0036139901390088.

        The algorithm solves the problem:
            min_Z (integral(laplacian(image_with_masked_pixels_replaced_by_Z)^2))
        or numerically:
            def loss(Z):
                img[mask] = Z
                return sum(square(
                           diff(img, 2, axis=0) + diff(img, 2, axis=1)
                          )) 
            img_inpainted = minimize(loss)

        This is done quickly by solving the equivalent problem of d(loss)/dZ = 0, with 
        values of the non-masked region as boundary condition. 

        The laplacian of an image at index i, j is numerically calculated as 
            L[i,j] = (  (image[i+1, j] - image[i, j]) - (image[i, j] - image[i-1, j])
                      + (image[i, j+1] - image[i, j]) - (image[i, j] - image[i, j-1])
                     )

        represented by a difference coefficient matrix
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0],
            ]

        For a particular Z_{i,j}, a to be determined value in the masked region, 
            d(sum(square(laplacian)))/dZ_{i,j} = sum_{k,l} 2 L[k, l] dL[k,l]/dZ_{i,j} = 0
        which yields a coefficient matrix 
            [[0,   0,   2,    0, 0],
             [0,   4,  -16,   4, 0],
             [2, -16,   40, -16, 2],
             [0,   4,  -16,   4, 0],
             [0,   0,    2,   0, 0],
            ]
        which when dotted with a corresponding region in the image imputed with Z should give 0. 

        (It can also be obtained by computing (dL/dZ).T . dL/dZ)

        This gives one equation per pixel to be imputed, as a row in a matrix corresponding to 
        the flattened version of the image.

        In addition, boundary conditions are added through rows specifying the value of known pixels.
        In total, that gives a number of equations (rows) equal to 
            sum(mask) + (image.size - sum(mask)) = image.size
        with columns corresponding to the pixels of the (flattened) image, also numbering image.size. 

        The resulting square matrix can then be uniquely solved.

        Methods
        -------
        inpaint(image: np.ndarray, mask: np.ndarray)
            performs inpainting of grayscale image in the regions specified by mask, which is
            an array of the same shape as image.
            Returns a new image the same size as the original.
        
    """

    def __init__(self, image_shape: Optional[tuple[int, int]] = None):
        """
        Parameters
        ----------
        image_shape : Optional[tuple[int, int]]
            Inpainter can be initialized with an image shape, so that it pre-computes the laplacian
            matrix, for use with many images of the same shape.
        """

        self.image_shape: Optional[tuple[int, int]]
        if image_shape is not None:
            self.initialize_image_shape(image_shape)
        else:
            self.image_shape = None

    def _generate_dSumSquareLaplacian_dZ_matrix(self, image_shape: tuple[int, int]):


        # Generate sparse matrix representing the Jacobian dL/dZ, where each row gives the Laplacian 
        # coefficients for image[i, j], with the columns corresponding to the flattened (full) image. 
        def to_flattened_index(i, j):
            return i * image_shape[1] + j
        m = 0
        row_ind = []
        col_ind = []
        data = []

        difference_coefficients = [((-1, 0), 1),
                                   ((+1, 0), 1),
                                   ((0, -1), 1),
                                   ((0, +1), 1),
                                   ((0,  0), -4),
                                  ]
        
        # Ignore first and last row, and first and last column, because the second difference can't be 
        # computed with the same coefficients. 
        # TODO: include second differences for border of image.
        for i in range(1, image_shape[0] - 1):
            for j in range(1, image_shape[1] - 1):
                # add coefficients for row m of Jacobian, which is the difference coefficients at 
                # location i, j in image, flattened. 
                for (di, dj), coeff in difference_coefficients:
                    row_ind.append(m)
                    col_ind.append(to_flattened_index(i+di, j+dj))
                    data.append(coeff)
                m += 1

        laplacian_jacobian = csc_matrix((data, (row_ind, col_ind)), shape=((image_shape[0] - 2) * (image_shape[1] - 2), image_shape[0] * image_shape[1]))

        # Finally, obtain the equations for d(sum(square(L)))/dZ == 0
        return 2 * laplacian_jacobian.T @ laplacian_jacobian

    def initialize_image_shape(self, image_shape: tuple[int, int]):
        """ Pre-computes the d(sum(square(L)))/dZ matrix, for use with many images of the same shape.

        """
        self.image_shape = image_shape
        self.dSumSquareLaplacian_dZ_matrix = self._generate_dSumSquareLaplacian_dZ_matrix(image_shape)

    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, shape_mismatch='raise') -> np.ndarray:
        """ Do the inpainting.
        
        Parameters
        ----------
        image : np.ndarray
            grayscale image
        mask : np.ndarray of bool
            array of same size as image with 1 for unknown (to be inpainted) pixels and 0 for known pixels.
        shape_mismatch : str
            what to do if the image shape is different from the initialized shape
                'raise': raise error
                'ignore': calculate dSumSquareLaplacian/dZ matrix for this new shape but don't replace the 
                          stored one.
                'replace': re-initialize this inpainter with image.shape

        Returns
        -------
        inpainted_image : np.ndarray

        """

        if self.image_shape is None:
            self.initialize_image_shape(image.shape)
            dSumSquareLaplacian_dZ_matrix = self.dSumSquareLaplacian_dZ_matrix

        elif image.shape != self.image_shape:
            if shape_mismatch == 'raise':
                raise ValueError(f"Image shape {image.shape} is not the same as initialized shape {self.image_shape}.")
            elif shape_mismatch == 'ignore':
                dSumSquareLaplacian_dZ_matrix = self._generate_dSumSquareLaplacian_dZ_matrix(image.shape)
            elif shape_mismatch == 'replace': 
                self.initialize_image_shape(image.shape)
                dSumSquareLaplacian_dZ_matrix = self.dSumSquareLaplacian_dZ_matrix
            else:
                raise ValueError(f"Unknown value for shape_mismatch: {shape_mismatch}.")
            
        else:
            dSumSquareLaplacian_dZ_matrix = self.dSumSquareLaplacian_dZ_matrix

        # combine 
        # - dSumSquareLaplacian/dZ == 0 equations,  for unknown pixels 
        # - Z_{i,j} == known_value,                 for known pixels
        # TODO: select only columns that have nonzero entries in the dSumSquareLaplacian/dZ == 0 equations
        A = sp_vstack([ dSumSquareLaplacian_dZ_matrix[mask.flatten(), :], 
                        sp_eye(image.size).tocsr()[~mask.flatten(), :] 
                     ])
        b = np.concatenate([ np.zeros(mask.sum()), 
                             image[~mask] 
                          ])

        image_inpainted = sp_solve(A, b).reshape(image.shape)

        return image_inpainted
