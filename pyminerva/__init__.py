# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.

__author__ = "jarvisNim in GitHub"
__version__ = "0.0.5"


from .base import (
    get_stock_history_by_fmp,
    get_stock_history_by_yfinance,
)

from .tech import (
    analyse_techs,
)

from .strategy import (
    sma_strategy,
    timing_strategy,
    get_vb_signals,
    show_vb_stategy_result,
    volatility_bollinger_strategy,
    get_reversal_signals,
    show_reversal_stategy_result,
    reversal_strategy,
    trend_following_strategy,
    control_chart_strategy,
    vb_genericAlgo_strategy,
    vb_genericAlgo_strategy2,
    gaSellHoldBuy_strategy,
)

