# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.


'''
이 곳에는 import 하는 것이 없는 순수한 데이터 자료론에 의한 데이터 상수값만 존재해야함.
기본 function 들은 base.py 로 가야함.
'''


api_key='0e836827495d195023016a96b5fe6e4a'
bok_key = 'OLSJAN6H7R43WEYUEV5Q'
fmp_key = 'f57bdcaa7d140c9de35806d47fbd2f91'

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
