'''
Prgram 명: Japan markets
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
목적: 일본 주식/채권/원자재/현금 데이터 분석
주요 내용
- 각 자산별 1년, 3년, 6년, 12년, 24년 마다의 평균값에서 +-시그마 범위를 벗어나는 경우 위험 표시
History
2024/01/25  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''
import pyminerva as mi
import pandas_ta as ta

from pykrx import stock
from pykrx import bond

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

TIMEFRAMES = ['1min', '1hour', '1day']





'''
Main Fuction
'''

if __name__ == "__main__":

    for x in WATCH_TICKERS['JP']:

        for asset, tickers in x.items():

            for ticker in tickers:

                if ticker == '':
                    continue

                # settings.py 에서 get_stock_history_by_fmp with timeframe 파일 만들어 줌.
                logger2.info('')                
                logger2.info(f' ##### {ticker}')

                df = mi.get_stock_history_by_fmp(ticker, TIMEFRAMES)
                if df.empty:  # fmp 에서 읽지 못하면 다음에는 yfinance 에서 읽도록 보완함. 
                    print(f'{ticker} df by fmp is empty')
                    df = mi.get_stock_history_by_yfinance(ticker, TIMEFRAMES)
                
                mi.timing_strategy(ticker, 20, 200) # 200일 이평 vs 20일 이평

                mi.volatility_bollinger_strategy(ticker, TIMEFRAMES) # 임계값 찾는 Generic Algorithm 보완했음.

                mi.vb_genericAlgo_strategy(ticker, TIMEFRAMES) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm       

                mi.vb_genericAlgo_strategy2(ticker, TIMEFRAMES) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm           

                mi.reversal_strategy(ticker, TIMEFRAMES) 

                mi.trend_following_strategy(ticker, TIMEFRAMES)  # 단기 매매 아님. 중장기 매매 기법, 1day 데이터만으로 실행

                mi.control_chart_strategy(ticker)

                mi.gaSellHoldBuy_strategy(ticker)

                print('=== End ===')
