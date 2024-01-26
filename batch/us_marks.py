'''
Prgram 명: 미국시장부문의 투자 전략
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
마켓은 주식/채권/원자재/현금등의 금융자산의 오픈된 시장을 의미하며, 
이 시장내에서의 다양한 개별적 전략(Strategy)들을 수립하고 이에 대한 백테스트 결과까지 도출하고 
주기적으로 이를 검증하며 매수/매도 기회를 포착하는 것을 목적으로 함. 
History
2023/11/16  Creat
20231119  https://github.com/crapher/medium 참조
          parameter 값을 최적화하기 위해서는 generic algorithm 을 사용하는 것을 default 로 정함.
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''
import yfinance as yf
import pandas_ta as ta
import pygad
import pygad.kerasga
import gym
import pyminerva as mi

from scipy import signal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gym import spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from math import sqrt, exp

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# gtta = {'VNQ':20, 'GLD':10, 'DBC':10, 'IEF':5, 'LQD':5, 'BNDX':5, 'TLT':5, \
#         'EEM':10, 'VEA':10, 'DWAS':5, 'SEIM':5, 'DFSV':5, 'DFLV':5}

TIMEFRAMES = ['1min', '1hour', '1day']


'''
1.10 Generic Algorithm & MACD Indicator Strategy
- https://medium.datadriveninvestor.com/my-approach-to-use-the-macd-indicator-in-the-market-part-2-3958aff26d0a
'''
def GaMacd_strategy():
    # Constants
    DEBUG = 0
    CASH = 10_000
    POPULATIONS = 30
    GENERATIONS = 50
    FILE_TRAIN = '../data/spy.train.csv.gz'
    FILE_TEST = '../data/spy.test.csv.gz'
    TREND_LEN = 7
    MIN_TRADES_PER_DAY = 1
    MAX_TRADES_PER_DAY = 10

    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    # Loading data, and split in train and test datasets
    def get_data():
        train = pd.read_csv(FILE_TRAIN, compression='gzip')
        train['date'] = pd.to_datetime(train['date'])
        train.ta.ppo(close=train['close'], append=True)
        train = train.dropna().reset_index(drop=True)

        test = pd.read_csv(FILE_TEST, compression='gzip')
        test['date'] = pd.to_datetime(test['date'])
        test.ta.ppo(close=test['close'], append=True)
        test = test.dropna().reset_index(drop=True)

        train = train[train['date'] > (test['date'].max() - pd.Timedelta(365 * 10, 'D'))]

        return train, test

    # Define fitness function to be used by the PyGAD instance
    def fitness_func(self, solution, sol_idx):
        
        try:
            # Get Reward from train data
            reward, wins, losses, pnl = get_result(train, train_dates,
                                        solution[           :TREND_LEN*1],
                                        solution[TREND_LEN*1:TREND_LEN*2],
                                        solution[TREND_LEN*2:TREND_LEN*3],
                                        solution[TREND_LEN*3:TREND_LEN*4])
        except ZeroDivisionError:
            # ZeroDivisionError가 발생한 경우 처리할 내용
            logger.error("Error: Division by zero!")
            logger.error(f'\n{reward:10.2f}, {pnl:10.2f}, {wins:6.0f}, {losses:6.0f}, {solution[TREND_LEN*1:TREND_LEN*2]}, {solution[TREND_LEN*3:TREND_LEN*4]}', end='')
        else:
            # 예외가 발생하지 않은 경우 실행할 내용
            pass

        if DEBUG:
            logger.debug(f'\n{reward:10.2f}, {pnl:10.2f}, {wins:6.0f}, {losses:6.0f}, {solution[TREND_LEN*1:TREND_LEN*2]}, {solution[TREND_LEN*3:TREND_LEN*4]}', end='')

        # Return the solution reward
        return reward
    







'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    0. 공통
    '''


    '''
    1. Stock
    '''

    for x in WATCH_TICKERS['US']:

        for asset, tickers in x.items():

            for ticker in tickers:

                if ticker == '':
                    continue           

                # settings.py 에서 get_stock_history_by_fmp with timeframe 파일 만들어 줌.
                logger2.info('')                
                logger2.info(f' ##### {ticker}')

                df = mi.get_stock_history_by_fmp(ticker, TIMEFRAMES)
                if df.empty:  # fmp 에서 읽지 못하면 다음에는 yfinance 에서 읽도록 보완함. 
                    logger2.error(f'{ticker} df by fmp is empty')

                    df = mi.get_stock_history_by_yfinance(ticker, TIMEFRAMES)
                    
                    logger2.info("dataframe from yfinance")
                    logger2.info(df.tail())
                
                mi.timing_strategy(ticker, 20, 200) # 200일 이평 vs 20일 이평

                mi.volatility_bollinger_strategy(ticker, TIMEFRAMES) # 임계값 찾는 Generic Algorithm 보완했음.

                mi.vb_genericAlgo_strategy(ticker, TIMEFRAMES) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm       

                mi.vb_genericAlgo_strategy2(ticker, TIMEFRAMES) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm           

                mi.reversal_strategy(ticker, TIMEFRAMES) 

                mi.trend_following_strategy(ticker, TIMEFRAMES)  # 단기 매매 아님. 중장기 매매 기법, 1day 데이터만으로 실행

                mi.control_chart_strategy(ticker)

                mi.gaSellHoldBuy_strategy(ticker)

                # GaMacd_strategy()

                print('=== End ===')

        

    # 2. Bonds
    # get_yields()