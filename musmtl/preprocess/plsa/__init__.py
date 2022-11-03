import numpy as np
from numpy import typing as npt
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from ._plsa import infer


class PLSA(TransformerMixin, BaseEstimator):
    """ Wrapper for plsa inference routine as sklearn's Transformer """

    dtype = np.float32  # currently we only support float32

    def __init__(
        self,
        n_topics: int = 10,
        n_iters: int = 20
    ):
        self.n_topics = n_topics
        self.n_iters = n_iters

    def fit_transform(
        self,
        X: sp.base.spmatrix
    ) -> npt.NDArray:
        """
        """
        X = X.tocoo().astype(self.dtype)  # transform the layout suitable for inference routine

        # initiate the weights
        loadings, self.components_ = self._init_weights(X)

        # infer the model!
        infer(
            X.row, X.col, X.data, loadings, self.components_,
            n_iters = self.n_iters
        )

        return loadings

    def fit(
        self,
        X: sp.base.spmatrix
    ):
        """
        """
        self.fit_transform(X)
        return self

    def _init_weights(
        self,
        X: sp.base.spmatrix
    ) -> tuple[npt.NDArray,
               npt.NDArray]:
        """
        """
        dtype = X.dtype
        n_rows, n_cols = X.shape
        H = np.random.rand(n_rows, self.n_topics).astype(dtype)
        H /= H.sum(1)[:, None]

        W = np.random.rand(n_cols, self.n_topics).astype(dtype)
        W /= W.sum(1)[:, None]
        return H, W
