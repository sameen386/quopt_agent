"""Real-data loading and the walk-forward window protocol.

Yahoo Finance terms do not permit redistribution of the raw price series, so
the repository ships the ticker universe and this download script, and caches
only the derived log-return matrix that the experiments consume.  The cache
makes every run after the first one fully offline and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_TICKERS", "load_returns", "download_prices",
           "walk_forward_windows", "Window"]

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM",
                   "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD", "MA", "CVX"]

CACHE_NAME = "returns_cache.csv"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Adjusted-close prices from Yahoo Finance via ``yfinance``."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required to download prices; install it with "
            "'pip install yfinance', or place a cached returns file in "
            "data/ and run with --offline") from exc

    frame = yf.download(tickers, start=start, end=end, auto_adjust=True,
                        progress=False, group_by="column")
    close = frame["Close"] if isinstance(frame.columns, pd.MultiIndex) \
        else frame
    close = close.reindex(columns=tickers)
    close = close.dropna(how="all").ffill().dropna()
    if close.empty:
        raise RuntimeError("Yahoo Finance returned no usable rows")
    return close


def load_returns(data_dir: Path, tickers: list[str] | None = None,
                 start: str = "2018-01-02", end: str = "2025-12-30",
                 offline: bool = False,
                 refresh: bool = False) -> tuple[np.ndarray, list[str],
                                                 pd.DatetimeIndex]:
    """Return ``(log_returns, tickers, dates)``, using the cache when possible.

    The cache is written on the first successful download so that results are
    reproducible even if the vendor revises its history.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / CACHE_NAME
    tickers = list(tickers or DEFAULT_TICKERS)

    if cache.exists() and not refresh:
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        frame = frame.reindex(columns=[t for t in tickers
                                       if t in frame.columns])
        return frame.to_numpy(dtype=float), list(frame.columns), frame.index

    if offline:
        raise FileNotFoundError(
            f"offline mode requested but no cache found at {cache}; run once "
            f"with network access to populate it")

    prices = download_prices(tickers, start, end)
    returns = np.diff(np.log(prices.to_numpy(dtype=float)), axis=0)
    frame = pd.DataFrame(returns, index=prices.index[1:],
                         columns=prices.columns)
    frame.to_csv(cache)
    return frame.to_numpy(dtype=float), list(frame.columns), frame.index


@dataclass
class Window:
    index: int
    train_start: int
    train_stop: int
    test_start: int
    test_stop: int


def walk_forward_windows(n_obs: int, n_windows: int = 20, lookback: int = 126,
                         horizon: int = 63) -> list[Window]:
    """Non-overlapping out-of-sample windows spanning the available history.

    The stride is chosen so that the requested number of windows fits the
    sample; each window estimates on ``lookback`` days and is evaluated on the
    following ``horizon`` days, which never overlap the estimation period.
    """
    span = lookback + horizon
    if n_obs < span:
        raise ValueError(f"need at least {span} observations, got {n_obs}")
    last_start = n_obs - span
    if n_windows == 1:
        starts = [last_start]
    else:
        starts = np.linspace(0, last_start, n_windows).round().astype(int)
    return [Window(index=i, train_start=int(s), train_stop=int(s + lookback),
                   test_start=int(s + lookback), test_stop=int(s + span))
            for i, s in enumerate(starts)]
