# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.

import sys, os
import pandas as pd
import requests
import yfinance as yf
import warnings
import logging, logging.config, logging.handlers

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from fredapi import Fred

from .utils import constant as cst


'''
공통 영역
'''
warnings.filterwarnings('ignore')

now = datetime.today()
global to_date, to_date2, to_date3
to_date = now.strftime('%d/%m/%Y')
to_date2 = now.strftime('%Y-%m-%d')
to_date3 = now.strftime('%Y%m%d')
# print('to_date: ', to_date)
# print('to_date2: ', to_date2)
# print('to_date3: ', to_date3)

global from_date_LT, from_date_MT, from_date_ST, from_date_LT2, from_date_MT2, from_date_ST2, from_date_LT3, from_date_MT3, from_date_ST3
# Used to analyze during 3 months for short term
_date = now + relativedelta(months=-3)
from_date_ST = _date.strftime('%d/%m/%Y')
from_date_ST2 = _date.strftime('%Y-%m-%d')
from_date_ST3 = _date.strftime('%Y%m%d')

# Used to analyze during 5 years for middle term (half of 10year Economic cycle)
_date = now + relativedelta(years=-5)
from_date_MT = _date.strftime('%d/%m/%Y')
from_date_MT2 = _date.strftime('%Y-%m-%d')
from_date_MT3 = _date.strftime('%Y%m%d')

# Used to analyze during 50 years for long term (5times of 10year Economic cycle)
_date = now + relativedelta(years=-50)
from_date_LT = _date.strftime('%d/%m/%Y') 
from_date_LT2 = _date.strftime('%Y-%m-%d')
from_date_LT3 = _date.strftime('%Y%m%d')

# print('Short: ' + from_date_ST + '   Middle: ' + from_date_MT + '    Long: ' + from_date_LT)


# create a logger with the name from the config file. 
# This logger now has StreamHandler with DEBUG Level and the specified format in the logging.conf file
logger = logging.getLogger('batch')
logger2 = logging.getLogger('report')


utils_dir = os.getcwd() + '/batch/Utils'
reports_dir = os.getcwd() + '/batch/reports'
data_dir = os.getcwd() + '/batch/reports/data'
database_dir = os.getcwd() + '/database'
batch_dir = os.getcwd() + '/batch'
sys.path.append(utils_dir)
sys.path.append(reports_dir)
sys.path.append(data_dir)
sys.path.append(database_dir)
sys.path.append(batch_dir)

fred = Fred(api_key=cst.api_key)

#####################################
# funtions
#####################################

# financial modeling 에서 stock hitory 가져와 csv 파일로 저장하기까지. 
def get_stock_history_by_fmp(ticker:str, periods:list):  # period: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day

    for period in periods:
        url = f'https://financialmodelingprep.com/api/v3/historical-chart/{period}/{ticker}?from={from_date_MT2}&to={to_date2}&apikey={cst.fmp_key}'
        try:
            buf = requests.get(url).json()
            df = pd.DataFrame(buf, columns=['date', 'open', 'low','high','close','volume'])
            if df.empty:
                return df
            df['ticker'] = ticker
            df.to_csv(data_dir + f'/{ticker}_hist_{period}.csv', index=False)
        except Exception as e:
            print('Exception: {}'.format(e))
        
    return df


# yahoo finance 에서 stock hitory 가져와 csv 파일로 저장하기까지. 단, 1day 만 가능. 
def get_stock_history_by_yfinance(ticker:str, timeframes:list):

    for timeframe in timeframes:
        try:
            if timeframe == '1min':
                _interval = "1m"                
                _period = "7d"  # yahoo: Only 7 days worth of 1m granularity data
            elif timeframe == '1hour':
                _interval = "1h"
                _period = "3mo"
            else:
                _interval = "1d"
                _period = "3y"

            df = yf.download(tickers=ticker, period=_period, interval=_interval)


            df = df.reset_index()
            if df.empty:
                return df
 
            df['ticker'] = ticker

            new_columns = ['date', 'open', 'high','low','close', 'adj close', 'volume', 'ticker']  # yfinance 에서는 column 명이 대문자.
            df.columns = new_columns

            df.to_csv(data_dir + f'/{ticker}_hist_{timeframe}.csv', index=False, mode='w')

        except Exception as e:
            print('Exception: {}'.format(e))
        
    return df    