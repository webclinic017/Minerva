# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from .utils import constant as cst
from .data.techs_data import (
    daily_returns,
    cumulative_returns,
    max_drawdown,
)

'''
공통 영역
'''
ta_list = ['mfi', 'ppo', 'bb', 'sma', 'rsi', 'vwap']



'''
funtions
'''

def analyse_techs(ticker=None):
    """
    This function retrieves all the available technical Analysis indexed on pandas_ta. 
    This function also allows the users to specify which technical anlysis do they want to draw graphic;
    so on, all the technical graphs will be returned.

    Args:
        ticker (:obj:`str`, optional): name of the ticker to analyse all its available technical analysis from.

    Returns:
        :obj:`pandas.DataFrame` - analysis:

                country | name | full_name | symbol | isin | asset_class | currency | stock_exchange | def_stock_exchange
                --------|------|-----------|--------|------|-------------|----------|----------------|--------------------
                xxxxxxx | xxxx | xxxxxxxxx | xxxxxx | xxxx | xxxxxxxxxxx | xxxxxxxx | xxxxxxxxxxxxxx | xxxxxxxxxxxxxxxxxx

    Raises:
        ValueError: raised when any of the input arguments is not valid.
        FileNotFoundError: raised when `etfs.csv` file was not found.
        IOError: raised when `etfs.csv` file is missing.

    """

    return None


def analyse_DrawDown(tickers:list):
    """
    Max DrawDown 그래프 생성
    Args:
        tickers(:obj:'list', optional) yahoo finance 기준 list of tickers
    Returns:
        :obj: 'file' - drawdown_{tickers}.png
    Raises:
        ValueError: raised when any of the input arguments is not valid.      
    """

    if tickers is not None and not isinstance(tickers, list): # input
        raise ValueError("ERR#0025: specified tickers value not valid.")
    
    
    threshold_value = -0.3
    plt.figure(figsize=(16,4*len(tickers)))
    for i, tick in enumerate(tickers):
        ticker = yf.Ticker(tick)
        prices = ticker.history(period='12y')['Close'] # 12: life cycle
        dret = daily_returns(prices)
        cret = cumulative_returns(dret)
        ddown = max_drawdown(cret)
        ddown[ddown.values < -0.3]

        plt.subplot(len(tickers), 1, i + 1)
        plt.grid()
        plt.bar(ddown.index, ddown, color='royalblue')
        plt.title(ticker)
        plt.axhline(y=threshold_value, color='red', linestyle='--', label='Threshold')
        plt.xlabel('Date')
        plt.ylabel('Draw Down %')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.show()
    # plt.savefig(f'drawdown_{tickers}.png')

    return ddown 






