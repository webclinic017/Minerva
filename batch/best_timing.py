'''
Prgram 명: best timing.py
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
1. 후보 종목 중 12년 평균 상단 3시그마 값 도달전 표시없이 skip,
   3시그마 도달하면 모니터링 시작(거래량, 기술적분석값, ...), 
   2시그마 도달시 매수 투자계획서 작성, 매수 결정시 분할 매수 시작
   1시그마 도달시 저항선이 될지, 지지선이 될지의 각각의 확률 표현, 저항선이면 매수분 매도, 지지선이며 추가 매수
   all-time-high 에서 거래량 + 기술적 분석을 통한 매도 시점 분석
2. 후보 종목 중 12년 평균 하단 3시그마 값 도달전 표시없이 skip,
   3시그마 도달하면 모니터링 시작(거래량, 기술적분석값, ...)
   2시그마 도달시 매수 투자계획서 작성, 매수 결정시 분할매수 시작
   1시그마 도달시 저항선이 될지,, 지지선이 될지의 각각의 확률 표현, 저항선이면 매수분 매도, 지지선이면 추가 매수
   all-time-low 에서 거래량 + 기술적 분석을 통한 매도 시점 분석
3. 보유 종목 중 매도 전략
   이동평균 하락 돌파시(중기, 장기, 단기제외) 거래량 + 기술적 분석을 통한 매도 계획 수립

History
20230120  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *


'''
0. 공통영역 설정
'''

import requests
import yfinance as yf
import pandas_ta as ta
from bs4 import BeautifulSoup as bs


# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info()
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
to_date_2 = pd.to_datetime(to_date2)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()