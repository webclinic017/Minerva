'''
Prgram 명: 미국 자산별 Business부문의 technical Analysis 만들기
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
국가별 Economic 사이클을 분석하고, 자산시장별 금융환경 분석 그리고 ETF를 통한 섹터별 투자기회를 탐색하는 목적임.  
- country: 잠재성장률대비 real GDP YoY, (include KR.Export) = nominal GDP - CPI
- market: Nasdaq, S&P500, KOSPI, KOSDAQ, US 3Y/10Y/20Y BOND, KR 1Y BOND, GOLD, OIL, COLLAR, YEN, WON, EURO 
- business: 각 섹터별 ETF
History
- 20231204  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

import requests
from bs4 import BeautifulSoup as bs
import yfinance as yf

'''
0. 공통영역 설정
'''
# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))