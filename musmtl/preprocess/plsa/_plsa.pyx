import cython

from cython cimport floating, integral

from cython.parallel import parallel, prange
from tqdm import tqdm

from libc.stdlib cimport free, malloc
from libc.string cimport memcpy
from libc.math cimport log, sqrt

import numpy as np
from numpy import typing as npt


def infer(
    row: npt.NDArray[np.int32],
    col: npt.NDArray[np.int32],
    val: npt.NDArray[np.float32],
    doc_topic: np.NDArray[np.float32],
    term_topic: np.NDArray[np.float32],
    n_iters: int,
    eps: float = 1e-12
) -> None:
    """
    """
    _infer(row, col, val, doc_topic, term_topic, n_iters)


@cython.cdivision(True)
@cython.boundscheck(False)
def _infer(integral[:] dt_row, integral[:] dt_col, floating[:] dt_val,
           floating[:, :] doc_topic, floating[:, :] term_topic,
           int n_iters, floating eps=1e-12):
    """
    """
    # declare some local variables
    dtype = np.float64 if floating is double else np.float32

    cdef integral n_docs, n_topics, n_terms, nnz = dt_val.shape[0]
    cdef integral i, idx, z, d, t
    cdef floating s, q

    # n_docs, n_topics = doc_topic.shape
    n_docs = doc_topic.shape[0]
    n_topics = doc_topic.shape[1]
    n_terms = term_topic.shape[0]

    cdef floating[:] p = np.zeros((n_topics,), dtype=dtype)  # buffer
    cdef floating[:] term_sum = np.zeros((n_topics,), dtype=dtype)
    cdef floating[:] doc_sum = np.zeros((n_docs,), dtype=dtype)
    cdef floating[:, :] topic_full = np.zeros((nnz, n_topics), dtype=dtype)

    for i in range(n_iters):
        ### Expectation ###
        for idx in range(nnz):
            d, t = dt_row[idx], dt_col[idx]
            s = eps  # sum to be used for the normalization
            for z in range(n_topics):
                p[z] = doc_topic[d, z] * term_topic[t, z]
                s += p[z]
            for z in range(n_topics):
                topic_full[idx, z] = p[z] / s
        ### Maximization ###
        doc_topic[:] = 0.
        term_topic[:] = 0.
        term_sum[:] = eps
        doc_sum[:] = eps
        for idx in range(nnz):
            for z in range(n_topics):
                q = dt_val[idx] * topic_full[idx, z]
                term_topic[dt_col[idx], z] += q
                term_sum[z] += q
                doc_topic[dt_row[idx], z] += q
                doc_sum[dt_row[idx]] += q
        # Normalize P(topic | doc)
        for d in range(n_docs):
            for z in range(n_topics):
                doc_topic[d, z] /= doc_sum[d]
        # Normalize P(term | topic)
        for z in range(n_topics):
           for t in range(n_terms):
               term_topic[t, z] /= term_sum[z]
