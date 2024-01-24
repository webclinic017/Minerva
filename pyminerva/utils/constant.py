# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.

import sys, os

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

TIMEFRAMES = ['1min', '1hour', '1day']

ASSETS = ['stock', 'bond', 'commodity', 'cash']
# 보유종목들
MY_TICKERS = {
    'US':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'KR':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],    
    'EU':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'JP':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'CN':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'US':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'DE':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],    
    'IN':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'SG':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
}

# 관심종목들
WATCH_TICKERS = {
    'US':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'KR':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],    
    'EU':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'JP':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'CN':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'US':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'DE':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],    
    'IN':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'SG':[{'bond':['TLT']}, {'stock':['SPY','QQQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
}