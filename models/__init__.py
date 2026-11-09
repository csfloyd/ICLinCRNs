"""
ICL Models package.

This package contains different model architectures for in-context learning.
"""

from .base_icl_model import BaseICLModel
from .markov_icl import MatrixTreeMarkovICL
from .polynomial_icl import RandomPolynomialICL
from .wta_icl import WinnerTakesAllICL
from .nonlinear_markov_icl import NonlinearMarkovICL
from .markov_icl_regression import MatrixTreeMarkovICLRegression
from .mlp_icl_regression import MLPICLRegression
from .markov_icl_oddball import MatrixTreeMarkovICLOddball
from .mlp_icl_oddball import MLPICLOddball
from .qk_icl import QKICL
from .mlp_icl import MLPICL
from .enzyme_cascade_icl import EnzymeCascadeICL

__all__ = [
    'BaseICLModel',
    'MatrixTreeMarkovICL',
    'RandomPolynomialICL',
    'WinnerTakesAllICL',
    'NonlinearMarkovICL',
    'MatrixTreeMarkovICLRegression',
    'MLPICLRegression',
    'MatrixTreeMarkovICLOddball',
    'MLPICLOddball',
    'QKICL',
    'MLPICL',
    'EnzymeCascadeICL'
]

