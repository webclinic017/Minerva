# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from .utils import constant as cst


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









