from typing import Union, Optional

import numpy as np
from numpy import typing as npt
from scipy import sparse as sp

from sklearn.base import TransformerMixin, BaseEstimator
from threadpoolctl import threadpool_limits
from implicit.als import AlternatingLeastSquares


class WMF(TransformerMixin, BaseEstimator):
    """ Wrapper for AlternatingLeastSquares as sklearn's Transformer """

    def __init__(
        self,
        factors: int = 50,
        regularization: float = 0.1,
        alpha: float = 1.,
        epsilon: float = 0.1,
        log_weight_transform: bool = False,
        dtype: npt.DTypeLike = np.float32,
        iterations: int = 20,
        num_threads: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.epsilon = epsilon
        self.dtype = dtype
        self.iterations = iterations
        self.num_threads = num_threads
        self.random_state = random_state
        self.log_weight_transform = log_weight_transform

    def fit_transform(
        self,
        X: sp.base.spmatrix
    ) -> npt.NDArray:
        """
        """
        # do the preproc
        self._weight_transform(X)

        with threadpool_limits(limits=1, user_api='blas'):
            mf = AlternatingLeastSquares(
                self.factors,
                self.regularization,
                1.,  # we process weight transform outside
                self.dtype,
                use_native = True,
                use_cg = True,
                use_gpu = False,
                iterations = self.iterations,
                num_threads = self.num_threads,
                random_state = self.random_state
            )
            mf.fit(X, show_progress=False)

        self.components_ = mf.item_factors
        return mf.user_factors

    def fit(
        self,
        X: sp.base.spmatrix
    ):
        """
        """
        self.fit_transform(X)
        return self

    def _weight_transform(
        self,
        X: sp.base.spmatrix
    ):
        """
        """
        if self.log_weight_transform:
            X.data[:] = 1. + self.alpha * np.log(1. + (X.data / self.epsilon))
        else:
            X.data[:] = 1. + self.alpha * X.data[:]
