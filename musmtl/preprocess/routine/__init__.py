from typing import Any, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from numpy import typing as npt
from scipy import sparse as sp
from sklearn.mixture import GaussianMixture

from ..plsa import PLSA


logger = logging.getLogger(__name__)


@dataclass
class InteractionData:
    mat: sp.base.spmatrix
    row_entities: list[Any]
    col_entities: list[Any]


@dataclass
class NumericData:
    values: npt.NDArray
    entities: list[Any]


@dataclass
class ProcessedData:
    row_factors: npt.NDArray
    row_entities: list[Any]
    col_factors: Optional[npt.NDArray] = None
    col_entities: Optional[list[Any]] = None


def process_factor_reduce(
    data: Union[InteractionData, NumericData],
    n_components: int = 50,
    n_iters: int = 100
) -> ProcessedData:
    """ preprocess the input data

    It processes the data (either :obj:`~musmtl.preprocess.routine.InteractionData`
    or :obj:`~musmtl.preprocess.routine.NumericData`) to get the factors
    per entity. In case of :obj:`~musmtl.preprocess.routine.InteractionData`,
    `pLSA`_, while `Gaussian mixture model`_ is employed for
    :obj:`~musmtl.preprocess.routine.NumericData`.

    Args:
        data: input data to be processed
        n_components: the number of components either of latent component model.
        n_iters: the number of inference updates for the batch. It only is
                 applied to the pLSA model.

    Returns:
        procssed data with factors and entity ids

    .. _pLSA:
        https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis
    .. _Gaussian mixture model:
        https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model
    """
    if isinstance(data, InteractionData):
        # do the routine with pLSA
        plsa = PLSA(n_components, n_iters)
        row_factors = plsa.fit_transform(data.mat)

        out = ProcessedData(
            row_factors = row_factors,
            col_factors = plsa.components_,
            row_entities = data.row_entities,
            col_entities = data.col_entities
        )

    else:
        # run GMM on the scalar numeric values
        gmm = GaussianMixture(n_components,
                              max_iter=n_iters)
        gmm.fit(data.values[:, None])
        row_factors = gmm.predict_proba(data.values[:, None])

        out = ProcessedData(
            row_factors = row_factors,
            row_entities = data.entities
        )

    return out



def fit_factors(
    config_fn: str,
    load_funcs: dict[str, Callable],
    required_data: dict[str, tuple[str, ...]],
    post_process_hooks: Optional[dict[str, tuple[Callable, str, ...]]] = None,
    n_components: int = 50,
    n_iters: int = 100,
) -> dict[str, ProcessedData]:
    """ fit factors for learning "learning targets" of MTL

    it fits various latent variables from the raw data given for further
    MTL procedure. Latent variables are density or simplex, so that we can
    apply KL-divergence as the learning objective further down in MTL.

    Args:
        config_fn: filename of configuration file where the filenames of
                   each `aspect` is saved.
        load_funcs: dictionary of callable with which the raw data of each
                    aspect is loadded
        required_data: dictionary of tuple of flags indicates which data
                       speicified in the configuration file is needed for
                       the load_func of corresponding aspect
        post_process_hooks: contains post processing functions per aspect
                            if needed.
        n_components:
            the number of components either of latent component model.
        n_iters: the number of inference updates for the batch. It only is
                 applied to the pLSA model.
        verbose: set verbosity

    Returns:
        processed factors per aspect in dictionary
    """
    # load the config
    with Path(config_fn).open('r') as fp:
        config = json.load(fp)

    factor_output = {}
    for task, load_func in load_funcs.items():
        logger.info(f'Processing {task}...')

        required = required_data[task]

        # if there's any missing required data in config, we skip
        if any(d not in config for d in required):
            continue

        data = load_func(*[config[d] for d in required])
        factor = process_factor_reduce(data,
                                       n_components=n_components,
                                       n_iters=n_iters)

        if post_process_hooks is not None and task in post_process_hooks:
            pp_func, pp_required = post_process_hooks[task]
            pp_args = [config[r] for r in pp_required]
            factor = pp_func(factor, *pp_args)

        factor_output[task] = {
            factor.row_entities[i]: factor.row_factors[i]
            for i in range(len(factor.row_entities))
        }

    return factor_output
