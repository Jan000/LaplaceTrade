# tests/test_features.py
"""Tests for the micro-structure feature engine, focused on look-ahead safety."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptotrader.data.features import MicrostructureFeatureEngine
from cryptotrader.data.ingestion import make_synthetic_ohlcv


def test_feature_names_and_shape() -> None:
    ohlcv = make_synthetic_ohlcv(n=500, seed=1)
    fe = MicrostructureFeatureEngine()
    feats = fe.transform(ohlcv)
    # transform returns the model features plus the "atr" helper column.
    assert list(feats.columns) == fe.feature_names + ["atr"]
    assert "atr" not in fe.feature_names
    assert len(feats) == len(ohlcv)


def test_no_lookahead_prefix_stability() -> None:
    """Feature row t must depend only on bars <= t.

    Truncating the input after bar t must not change any feature value at t.
    """
    ohlcv = make_synthetic_ohlcv(n=600, seed=2)
    fe = MicrostructureFeatureEngine()
    full = fe.transform(ohlcv)

    cut = 400
    truncated = fe.transform(ohlcv.iloc[: cut + 1])
    pd.testing.assert_series_equal(
        full.iloc[cut], truncated.iloc[cut], check_names=False
    )


def test_incremental_matches_batch() -> None:
    """The live incremental path must equal the vectorised batch path."""
    ohlcv = make_synthetic_ohlcv(n=400, seed=3)
    fe_batch = MicrostructureFeatureEngine()
    batch = fe_batch.transform(ohlcv)

    fe_live = MicrostructureFeatureEngine()
    from cryptotrader.core.types import Bar

    last_idx = len(ohlcv) - 1
    incremental = None
    for ts, row in ohlcv.iterrows():
        incremental = fe_live.update(
            Bar(ts.to_pydatetime(), row.open, row.high, row.low, row.close, row.volume)
        )

    assert incremental is not None
    np.testing.assert_allclose(
        incremental.to_numpy(),
        batch.iloc[last_idx].to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )
