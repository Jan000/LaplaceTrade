# src/cryptotrader/ml/__init__.py
"""ML engine: labeling, model wrappers, and a rule-based baseline predictor."""

from cryptotrader.ml.model import (
    LightGBMPredictor,
    MomentumBaselinePredictor,
    make_triple_barrier_labels,
)

__all__ = [
    "LightGBMPredictor",
    "MomentumBaselinePredictor",
    "make_triple_barrier_labels",
]
