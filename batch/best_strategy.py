'''
Prgram 명: best strategy.py
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
현재 투자전략 다음의 전략을 수립하고 검증하여 최종 투자실행하도록 하는 프로그램으로,
- 투자계획서: 투자대상, 목표(수익률), 실행전략(규모, 진입시점, 탈출시점, loss cut),  위기관리방안(돌발상황)
- 하이일드채권
- 이머징채권
History
20230104  Create
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
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
to_date_2 = pd.to_datetime(to_date2)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()

high_yields = ['JNK', 'LQD', 'HYG']

'''
1. 미국 하이일드 채권
- 국채투자 다음 투자후부로 검토중
- 미국 경기침체로 인한 경기가 바닥을 치고 상승하는 시기 투자대상으로 선정
  . 투자시기: 경기 침체 발생으로 국채가 최고가를 갱신하고, 투자위험등급의 회사채 중 하이일드 ETF 는  최저점을 통과하는 시점
  . 투자종목: JNK, LQD, HYG
  . 투자전략: rsi stg, trend following stg, panic volume stg, max drawdown stg, 
  . 수익률:
  . 규모:
  . 진입시점
  . 탈출시점
  . loss cut
  . 위기관리 방안
- 한국 하이일드 채권은 검토 대상아님, (단, 국가부도 위기상황에서는 투자 가능)
'''
class HighYield():

    def __init__(self):
        pass

    def get_data(ticker, window):

        ticker = yf.Ticker(ticker)
        df = ticker.history(period='36mo') # test: 10mo, real: 36mo
        df['feature'] = signal.detrend(df['Close'])
        df['mean'] = df['feature'].rolling(window=window).mean()    
        df['std'] = df['feature'].rolling(window=window).std()
        
        return df


    def rsi_stg(self, ticker):
        df[tech_type] = df.ta.rsi(14)
        pass


    # Momentum.Relative Strength Index (RSI)
    def rsi(ticker, df, slope, tech_type, sbubplot_cnt, idx):    
        df['rsi'] = df.ta.rsi(14)
        make_plot(ticker, df, slope, tech_type, sbubplot_cnt, idx)
        
        return df, latest_date


    def trend_follow_stg(self, ):
        pass

    def panic_volume_stg(self, ):
        pass

    def max_drawdown_stg(self, ):
        pass
    
    def cal_return(self, ):
        pass

    def get_in(self, ):
        pass

    def get_out(self, ):
        pass

    def loss_cut(self, ):
        pass

    def beting(self, ):
        pass


hy = HighYield()
hy.rsi_stg()

