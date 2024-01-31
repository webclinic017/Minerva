# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.


'''
https://github.com/crapher/medium 참조
'''

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import pygad
import pygad.kerasga
import gym

from datetime import date, datetime, timedelta
from fredapi import Fred
from gym import spaces
from tqdm import tqdm
from scipy import signal
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from .utils import constant as cst
from .utils.strategy_funcs import (
    find_30days_ago,
)
from . import base

'''
공통 영역
'''

# logging
base.logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
base.logger2.info('')
base.logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

#####################################
# funtions
#####################################

'''
Timing Model & GTTA (Global Tactical Asset Allocation) Strategy
Asset Class Trend Following¶
http://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461
BUY RULE: Buy when monthly price > 10-month SMA.
SELL RULE: Sell and move to cash when monthly price < 10-month SMA.
GTAA consists of five global asset classes: US stocks, foreign stocks, bonds, real estate and commodities.
'''
# 새로운 포트폴리오 구성하는 방안으로 설정하면.
def sma_strategy(tick:str, short_sma=20, long_sma=200):
    data = pd.DataFrame()

    #Download ticker price data from yfinance
    ticker = yf.Ticker(tick)
    buf = ticker.history(period='36mo') # test: 10mo, real: 36mo
    #Calculate 10 and 20 days moving averages
    sma20 = buf.ta.sma(short_sma, append=True)
    sma200 = buf.ta.sma(long_sma, append=True)
    buf.ta.rsi(close="Close", length=14, append=True)
    #Create a column with buy and sell signals
    buf['Ticker'] = ticker
    buf['Signal'] = 0.0
    try: 
        buf['Signal'] = sma20 - sma200
    except TypeError:  # 530107.KS 히스토리가 1 레코드 밖에 없음.
        base.logger.error(f'Type Error: {tick}')
        raise Exception(f'Type Error: {tick}')
    buf['Pivot'] = np.where((buf['Signal'].shift(1)*buf['Signal']) < 0, 1, 0)  # 1로 되는 일자부터 매수 또는 매도후 현금
    data = pd.concat([data, buf])
        
    return data


def timing_strategy(ticker, short_sma, long_sma):

    try:
        result = sma_strategy(ticker, short_sma, long_sma)
    except:  # 530107.KS 히스토리가 1 레코드 밖에 없음.
        return
    buf = result[result['Pivot'] == 1].reset_index()
    # 날짜를 기준으로 최대 날짜의 인덱스를 찾기
    latest_indices = buf.groupby('Ticker')['Date'].idxmax()
    # 최대 날짜의 거래 내역을 발췌
    latest_records = buf.loc[latest_indices]
    # Change rate 비율만큼 Buy/Sell 실행할것, 초기 설정은 임계값 상승돌파하면 75% 추가매수, 하락돌파하면 75% 매도
    # pivot_tickers = latest_records[latest_records['Date']  >= day5_ago]  # for test: '2023-05-16'
    day30_ago = find_30days_ago()  # 30일 이전까지 피벗날짜 없으면 실행하기 늦었다고 판단했음.
    pivot_tickers = latest_records[latest_records['Date']  >= day30_ago]  # for test: '2023-05-16'

    base.logger2.info(f' Timing_strategy: {ticker} Only 1day '.center(60, '*'))

    if pivot_tickers.empty:
        base.logger2.info('***** 30일 이내 피봇 전환일자 없음.')
    else:
        pivot_tickers['Change_rate'] = np.where((pivot_tickers['Signal']) > 0, 1.75, 0.25)
        change_date = pivot_tickers['Date'].values
        change_rate = float(pivot_tickers['Change_rate'].values) * 100
        base.logger2.info(f'##### {long_sma}일 이동평균과 {short_sma}일 이동평균: Timing Strategy 에 따라 {change_date} 일부터 비중 {change_rate} % 로 조정할 것 !!! #####')
        base.logger2.info(pivot_tickers)
        # 검증용 백데이터 제공
        tick = pivot_tickers['Ticker']
        df = pd.DataFrame()
        for t in tick:
            buf = result[result['Ticker'] == t].tail(3)
            df = pd.concat([df, buf])
        base.logger2.info(df) # 검증시 사용


'''
Volatility-Bollinger Bands Strategy
Using this method, you can obtain buy and sell signals determined by the selected strategy.
The resulting signals are represented as a series of numerical values:
  '1' indicating a buy signal,
  '0' indicating a hold signal, and
  '-1' indicating a sell signal
'''
def get_vb_signals(df):
    pd.options.mode.chained_assignment = None
    df.ta.bbands(close=df['close'], length=20, append=True)   
    df = df.dropna()
    df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
    df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
    df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
    df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1
    min_volatility = df['volatility'].mean() - df['volatility'].std()
    # Buy Signals
    df['signal'] = np.where((df['volatility'] > min_volatility) & (df['close_percentage'] < 0.25), 1, 0)
    # Sell Signals
    df['signal'] = np.where((df['close_percentage'] > 0.75), -1, df['signal'])

    return df['signal']

def show_vb_stategy_result(timeframe, df):
    if df.empty:
        return None
    waiting_for_close = False
    open_price = 0
    profit = 0.0
    wins = 0
    losses = 0
    for i in range(len(df)):
        signal = df.iloc[i]['signal']
        ticker = df.iloc[i]['ticker']
        if signal == 1 and not waiting_for_close:
            waiting_for_close = True
            open_price = df.iloc[i]['close']
        elif (signal == -1 and waiting_for_close):
            waiting_for_close = False
            close_price = df.iloc[i]['close']
            profit += close_price - open_price
            wins = wins + (1 if (close_price - open_price) > 0 else 0)
            losses = losses + (1 if (close_price - open_price) < 0 else 0)

    try:
        win_rate = (wins/(wins + losses) if wins + losses > 0 else 0) * 100
        if win_rate >= 80 and profit > 1200000:
            base.logger2.info(f'********** Volatility-Bollinger Bands Strategy: Result of {ticker} - {timeframe} Timeframe '.center(60, '*'))
            base.logger2.info(f'* Profit/Loss: {profit:.2f}')
            base.logger2.info(f"* Wins: {wins} - Losses: {losses}")        
            base.logger2.info(f"* Win Rate: {win_rate:6.2f}%")
        else:
            pass
    except Exception as e:
        base.logger.error(' >>> Exception1: {}'.format(e))  


def volatility_bollinger_strategy(ticker:str, TIMEFRAMES:list):
    # Iterate over each timeframe, apply the strategy and show the result
    for timeframe in TIMEFRAMES:
        df = pd.read_csv(base.data_dir + f'/{ticker}_hist_{timeframe}.csv')
        if df.empty:
            continue
        # Add the signals to each row
        try:
            df['signal'] = get_vb_signals(df)
            base.logger2.info(f" volatility_bollinger_strategy: {ticker} / {timeframe} ".center(60, "*"))            
        except KeyError as e: # 히스토리 레코드가 1건이라 볼린저밴드 20 을 만들수 없음.
            base.logger.error(f"volatility_bollinger_strategy Key Error ({ticker} / {timeframe}): {e}")
            base.logger2.error(f"volatility_bollinger_strategy Key Error ({ticker} / {timeframe}): {e}")
            df['signal'] = 0
        df2 = df[df['ticker'] == ticker]
        # Get the result of the strategy
        show_vb_stategy_result(timeframe, df2)


'''
Reversal Strategy
aims to identify potential trend reversals in stock prices
'''
def get_reversal_signals(df):
    # Buy Signals
    df['signal'] = np.where((df['low'] < df['low'].shift()) & (df['close'] > df['high'].shift()) & (df['open'] < df['close'].shift()), 1, 0)
    # Sell Signals
    df['signal'] = np.where((df['high'] > df['high'].shift()) & (df['close'] < df['low'].shift()) & (df['open'] > df['open'].shift()), -1, df['signal'])

    return df['signal']

def show_reversal_stategy_result(timeframe, df):
    if df.empty:
        return None
    waiting_for_close = False
    open_price = 0
    profit = 0.0
    wins = 0
    losses = 0

    for i in range(len(df)):
        signal = df.iloc[i]['signal']
        ticker = df.iloc[i]['ticker']
        if signal == 1 and not waiting_for_close:
            waiting_for_close = True
            open_price = df.iloc[i]['close']
        elif signal == -1 and waiting_for_close:
            waiting_for_close = False
            close_price = df.iloc[i]['close']
            profit += close_price - open_price
            wins = wins + (1 if (close_price - open_price) > 0 else 0)
            losses = losses + (1 if (close_price - open_price) < 0 else 0)

    try:
        win_rate = ((wins/(wins + losses)) if wins + losses > 0 else 0) * 100
        if win_rate >= 80 and profit > 1200000:
            base.logger2.info(f'********** Reversal Strategy: Result of {ticker} for {timeframe} Timeframe '.center(60, '*'))
            base.logger2.info(f'* Profit/Loss: {profit:.2f}')
            base.logger2.info(f"* Wins: {wins} - Losses: {losses}")        
            base.logger2.info(f"* Win Rate: {win_rate:6.2f}%")  # if wins + losses == 0
        else:
            pass
    except Exception as e:
        base.logger.error(' >>> Exception2: {}'.format(e))

def reversal_strategy(ticker:str, TIMEFRAMES:list):
    # Iterate over each timeframe, apply the strategy and show the result
    for timeframe in TIMEFRAMES:    
        df = pd.read_csv(base.data_dir + f'/{ticker}_hist_{timeframe}.csv')

        # Add the signals to each row
        try:
            df['signal'] = get_reversal_signals(df)
            df2 = df[df['ticker'] == ticker]
            base.logger2.info(f" reversal_strategy: {ticker} / {timeframe} ".center(60, "*"))            
        except KeyError as e: # 히스토리 레코드가 1건이라 볼린저밴드 20 을 만들수 없음.
            base.logger.error(f"reversal_strategy Key Error ({ticker} / {timeframe}): {e}")
            base.logger2.error(f"reversal_strategy Key Error ({ticker} / {timeframe}): {e}")
            df['signal'] = 0
      
        # Get the result of the strategy
        show_reversal_stategy_result(timeframe, df2)


'''
Trend Following Strategy
Whether the market is experiencing a bull run or a bearish downturn, 
the goal is to hop on the trend early and stay on 
until there is a clear indication that the trend has reversed.
https://wire.insiderfinance.io/navigating-financial-markets-with-the-trend-following-strategy-ec02474169ba
'''
def trend_following_strategy(ticker:str, TIMEFRAMES:list):
    # Constants
    CASH = 1_000_000               # Cash in account, 달러, 엔화, 위안화, 유로, 원화 다 고려해서 100만으로...
    STOP_LOSS_PERC = -2.0        # Maximum allowed loss
    TRAILING_STOP = -1.0         # Value percentage for trailing_stop to lock in profits 
    TRAILING_STOP_TRIGGER = 2.0  # Percentage to start using the trailing_stop to "protect" earnings
    GREEN_BARS_TO_OPEN = 4       # Green bars required to open a new position

    for timeframe in TIMEFRAMES:   
        file_name = base.data_dir + f'/{ticker}_hist_{timeframe}.csv'        
        df = pd.read_csv(file_name)
        df = df.copy().reset_index(drop=True)
        df = df[::-1]  # 시작일자부터 BUY/SELL 를 정해서 계산해 올라와야 맞을듯. 

        df['date'] = pd.to_datetime(df['date'])
        # Calculate consecutive bars in the same direction
        df['bar_count'] = ((df['open'] < df['close']) != (df['open'].shift() < df['close'].shift())).cumsum()
        df['bar_count'] = df.groupby(['bar_count'])['bar_count'].cumcount() + 1
        df['bar_count'] = df['bar_count'] * np.where(df['open'].values < df['close'].values,1,-1)
        base.logger2.info(f" Trend Following Strategy: {ticker} / {timeframe} ".center(60, "*"))        

        # Variables Initialization
        cash = CASH
        shares = 0
        last_bar = None
        operation_last = 'WAIT'
        ts_trigger = 0
        sl_price = 0
        operation_last_old = None
        buf = []  

        # Generate operations
        for index, row in df.iterrows():
            date = row['date']
            # If there is no operation
            if operation_last == 'WAIT':
                if row['close'] == 0:
                    continue
                if last_bar is None:
                    last_bar = row
                    continue

                if row['bar_count'] >= GREEN_BARS_TO_OPEN:  # Identifying Trends
                    operation_last = 'BUY'
                    open_price = row['close']
                    ts_trigger = open_price * (1 + (TRAILING_STOP_TRIGGER / 100))  # Trailing Stops and Exit Points
                    sl_price = open_price * (1 + (STOP_LOSS_PERC / 100))  # Setting Stop-Loss
                    shares = int(cash // open_price)
                    cash -= shares * open_price
                else:
                    last_bar = None
                    continue     

            # If the last operation was a purchase
            elif operation_last == 'BUY':
                if row['close'] < sl_price:
                    operation_last = 'WAIT'
                    cash += shares * row['close']
                    shares = 0
                    open_price = 0
                    ts_trigger = 0
                    sl_price = 0
                elif open_price < row['close']:
                    if row['close'] > ts_trigger:
                        sl_price_tmp = row['close'] * (1 + (TRAILING_STOP / 100))
                        if sl_price_tmp > sl_price:
                            sl_price = sl_price_tmp

            if (operation_last != operation_last_old):
                temp = f"{date}: {operation_last:<5}: {round(open_price, 2):,} - Cash: {round(cash, 2):,} - Shares: {shares:,} - CURR PRICE: {round(row['close'],2):,} ({index}) - CURR POS: {round(shares * row['close'],2):,}"
                buf.append(temp)
            # base.logger2.info(f"{date}: {operation_last:<5}: {round(open_price, 2):8} - Cash: {round(cash, 2):8} - Shares: {shares:4} - CURR PRICE: {round(row['close'], 2):8} ({index}) - CURR POS: {round(shares * row['close'], 2)}")

            operation_last_old = operation_last
                            
            # if timeframe == '1day':  # @@@
            #     start_date = (datetime.now() - timedelta(days=90)).date().strftime('%Y-%m-%d')  # 3개월전 트랜젝션부터 보여주기
            #     if (operation_last != operation_last_old) and (date >= pd.to_datetime(start_date)):
            #         base.logger2.info(f"{date}: {operation_last:<5}: {round(open_price, 2):8} - Cash: {round(cash, 2):8} - Shares: {shares:4} - CURR PRICE: {round(row['close'], 2):8} ({index}) - CURR POS: {round(shares * row['close'], 2)}")
            #     operation_last_old = operation_last
            
            last_bar = row

        if shares > 0:
            cash += shares * last_bar['close']
            shares = 0
            open_price = 0

        if cash > 1200000:
            base.logger2.info("")
            base.logger2.info(f" ***** Cash after Trade {ticker} / {timeframe}: {round(cash, 2):,}")
            for b in buf:
                base.logger2.info(b)


'''
ControlChartStrategy
https://wire.insiderfinance.io/trading-the-stock-market-in-an-unconventional-way-using-control-charts-f6e9aca3d8a0
these seven rules proposed by Mark Allen Durivage
Rule 1 — One Point Beyond the 3σ Control Limit
Rule 2 — Eight or More Points on One Side of the Centerline Without Crossing
Rule 3 — Four out of five points in zone B or beyond
Rule 4 — Six Points or More in a Row Steadily Increasing or Decreasing
Rule 5 — Two out of three points in zone A
Rule 6–14 Points in a Row Alternating Up and Down
Rule 7 — Any noticeable/predictable pattern, cycle, or trend
'''
def control_chart_strategy(ticker):
    # Constants  
    ticker_file = base.data_dir + f'/{ticker}.csv'
    default_window = 10
    CASH = 1_000_000
    DEFAULT_WINDOW = 10
    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    def get_data(ticker, ticker_file):

        try:
            df = pd.read_csv(ticker_file)
        except:
            base.logger.info(f"Read csv file Not found -> yfinance making... : {ticker_file}")
            ticker = yf.Ticker(ticker)
            df = ticker.history(period='36mo')
            if len(df) <= 0:
                base.logger.error(f'ticker not found: {ticker}')
                return None
            
            df = df.reset_index()
            df['Date'] = df['Date'].dt.date
            df = df[['Date','Close']]
            df.columns = ['date', 'close']
            if len(df) > 0: df.to_csv(ticker_file, index=False)

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').resample('5T').agg('last')
        df = df.dropna()
        df['feature'] = signal.detrend(df['close'])
        return df.reset_index(drop=True)
    
    # Show result based on the selected rule
    def show_result(df, signal_field:str):  # signal_field: column 명 가르킴.
        # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
        ops = df[df[signal_field] != 0]
        ops = ops[ops[signal_field] != ops[signal_field].shift()]
        if (len(ops) > 0) and (ops.iat[0, -1] == -1): ops = ops.iloc[1:]
        if (len(ops) > 0) and (ops.iat[-1, -1] == 1): ops = ops.iloc[:-1]
        # Calculate P&L / operation
        ops['pnl'] = np.where(ops[signal_field] == -1, (ops['close'] - ops['close'].shift()) * (CASH // ops['close'].shift()), 0)
        # Calculate total P&L, wins, and losses
        pnl = ops['pnl'].sum()
        wins = len(ops[ops['pnl'] > 0])
        losses = len(ops[ops['pnl'] < 0])

        # logger2.info 정보가 너무 많아 TEST 결과 승률이 80% 이상인 경우만 display 하기 위하여 일부 display 순서 변경 20240122
        try:
            win_rate = ((wins/(wins + losses)) if wins + losses > 0 else 0) * 100
            if win_rate >= 80 and pnl > 1200000:
                # Show Result
                if signal_field == 'rule1':
                    base.logger2.info('Rule 1 — One Point Beyond the 3σ Control Limit')
                elif signal_field == 'rule2':
                    base.logger2.info('Rule 2 — Eight or More Points on One Side of the Centerline Without Crossing')
                elif signal_field == 'rule3':
                    base.logger2.info('Rule 3 — Four out of five points in zone B or beyond')
                elif signal_field == 'rule4':
                    base.logger2.info('Rule 4 — Six Points or More in a Row Steadily Increasing or Decreasing')
                elif signal_field == 'rul5':
                    base.logger2.info('Rule 5 — Two out of three points in zone A')
                elif signal_field == 'rule6':
                    base.logger2.info('Rule 6 – 14 Points in a Row Alternating Up and Down')
                else:
                    base.logger.error('control_chart_strategy rule number is not found.')

                base.logger2.info(f' Result of {ticker} for ({signal_field}) '.center(60, '*'))
                base.logger2.info(f"* Profit / Loss  : {pnl:.2f}")
                base.logger2.info(f"* Wins / Losses  : {wins} / {losses}")
                base.logger2.info(f"* Win Rate       : {win_rate:.2f}%")
            else:
                pass
        except Exception as e:
            base.logger.error(' >>> Exception3: {}'.format(e))

    # Rules definition
    def apply_rule_1(df, window = DEFAULT_WINDOW):
        # One point beyond the 3 stdev control limit
        df['sma'] = df['feature'].rolling(window=window).mean()
        df['3std'] = 3 * df['feature'].rolling(window=window).std()
        df['rule1'] = np.where(df['feature'] < df['sma'] - df['3std'], 1, 0)
        df['rule1'] = np.where(df['feature'] > df['sma'] - df['3std'], -1, df['rule1'])
        return df.drop(['sma','3std'], axis=1)

    def apply_rule_2(df, window = DEFAULT_WINDOW):
        # Eight or more points on one side of the centerline without crossing
        df['sma'] = df['feature'].rolling(window=window).mean()
        for side in ['upper', 'lower']:
            df['count_' + side] = (df['feature'] > df['sma']) if side == 'upper' else (df['feature'] < df['sma'])
            df['count_' + side] = df['count_' + side].astype(int)
            df['count_' + side] = df['count_' + side].rolling(window=8).sum()
        df['rule2'] = np.where(df['count_upper'] >= 8, 1, 0)
        df['rule2'] = np.where(df['count_lower'] >= 8, -1, df['rule2'])
        return df.drop(['sma','count_upper','count_lower'], axis=1)

    def apply_rule_3(df, window = DEFAULT_WINDOW):
        # Four out of five points over 1 stdev or under -1 stdev
        df['sma'] = df['feature'].rolling(window=window).mean()
        df['1std'] = df['feature'].rolling(window=window).std()
        df['rule3'] = np.where((df['feature'] < df['sma'] - df['1std']).rolling(window=5).sum() >= 4, 1, 0)
        df['rule3'] = np.where((df['feature'] > df['sma'] + df['1std']).rolling(window=5).sum() >= 4, -1, df['rule3'])
        return df.drop(['sma','1std'], axis=1)

    def apply_rule_4(df):
        # Six points or more in a row steadily increasing or decreasing
        df['rule4'] = np.where((df['feature'] < df['feature'].shift(1)) &
                            (df['feature'].shift(1) < df['feature'].shift(2)) &
                            (df['feature'].shift(2) < df['feature'].shift(3)) &
                            (df['feature'].shift(3) < df['feature'].shift(4)) &
                            (df['feature'].shift(4) < df['feature'].shift(5)), 1, 0)
        df['rule4'] = np.where((df['feature'] > df['feature'].shift(1)) &
                            (df['feature'].shift(1) > df['feature'].shift(2)) &
                            (df['feature'].shift(2) > df['feature'].shift(3)) &
                            (df['feature'].shift(3) > df['feature'].shift(4)) &
                            (df['feature'].shift(4) > df['feature'].shift(5)), -1, df['rule4'])
        return df

    def apply_rule_5(df, window = DEFAULT_WINDOW):
        # Two out of three points over 2 stdev or under -2 stdev
        df['sma'] = df['feature'].rolling(window=window).mean()
        df['2std'] = 2 * df['feature'].rolling(window=window).std()
        df['rule5'] = np.where((df['feature'] < df['sma'] - df['2std']).rolling(window=3).sum() >= 2, 1, 0)
        df['rule5'] = np.where((df['feature'] > df['sma'] + df['2std']).rolling(window=3).sum() >= 2, -1, df['rule5'])
        return df.drop(['sma','2std'], axis=1)

    def apply_rule_6(df, window = DEFAULT_WINDOW):
        # 14 points in a row alternating up and down
        df['sma'] = df['feature'].rolling(window=window).mean()
        df['1std'] = df['feature'].rolling(window=window).std()
        df['2std'] = 2 * df['1std']
        # Determine the zones for each row
        df['zone'] = None
        df.loc[df['feature'] > df['sma'], 'zone'] = '+C'
        df.loc[df['feature'] > df['sma'] + df['1std'], 'zone'] = '+B'
        df.loc[df['feature'] > df['sma'] + df['2std'], 'zone'] = '+A'
        df.loc[df['feature'] < df['sma'], 'zone'] = '-C'
        df.loc[df['feature'] < df['sma'] - df['1std'], 'zone'] = '-B'
        df.loc[df['feature'] < df['sma'] - df['2std'], 'zone'] = '-A'
        df['rule6'] = np.where((df['zone'] != df['zone'].shift()).rolling(window=14).sum() >= 14, 1, -1)
        return df.drop(['sma','1std','2std','zone'], axis=1)

    df = get_data(ticker, ticker_file)
    
    df = apply_rule_1(df)    
    show_result(df, 'rule1')

    df = apply_rule_2(df)
    show_result(df, 'rule2')

    df = apply_rule_3(df)
    show_result(df, 'rule3')

    df = apply_rule_4(df)
    show_result(df, 'rule4')

    df = apply_rule_5(df)
    show_result(df, 'rule5')

    df = apply_rule_6(df)
    show_result(df, 'rule6')


'''
Volatility & Bollinger Band with Generic Algorithm Strategy
'''
def vb_genericAlgo_strategy(ticker:str, TIMEFRAMES:list):
    # Constants
    POPULATIONS = 20
    GENERATIONS = 50
    CASH = 1_000_000

    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    # Loading data, and split in train and test datasets
    def get_data(timeframe):

        df = pd.read_csv(base.data_dir + f'/{ticker}_hist_{timeframe}.csv')
        if df.empty:
            base.logger.error(f'Read csv file {ticker} / {timeframe} is Empty')
            return None
        else:            
            df.ta.bbands(close=df['close'], length=20, append=True)
            df = df.dropna()
            df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
            df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
            df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
            df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1

            if (timeframe == '1min') or (timeframe == '1hour'):
                train, test = train_test_split(df, test_size=0.25, random_state=1104)
            else:
                _date = (datetime.now() - timedelta(days=365)).date().strftime('%Y-%m-%d')
                train = df[df['date'] < _date]
                test = df[df['date'] >= _date]

        return train, test, df

    
    # Define fitness function to be used by the PyGAD instance
    def fitness_func(self, solution, sol_idx):
        try:
            # total reward 가 최대값을 갖을 수 있는 solution[0],[1],[2] 의 변수들을 찾아서 최적화(=> pygad.GA()를 통해서)
            total_reward, _, _ = get_result(train, solution[0], solution[1], solution[2])
        except:
            reward = 0
            pass
        # Return the solution reward
        return total_reward

    # Define a reward function
    def get_result(df, min_volatility, max_buy_pct, min_sell_pct):
        # Generate a copy to avoid changing the original data
        df = df.copy().reset_index(drop=True)

        # Buy Signal
        df['signal'] = np.where((df['volatility'] > min_volatility) & (df['close_percentage'] < max_buy_pct), 1, 0)
        # Sell Signal
        df['signal'] = np.where((df['close_percentage'] > min_sell_pct), -1, df['signal'])

        # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
        result = df[df['signal'] != 0]
        result = result[result['signal'] != result['signal'].shift()]
        if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
        if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

        # Calculate the reward / operation
        result['total_reward'] = np.where(result['signal'] == -1, result['close'] - result['close'].shift(), 0)

        # Generate the result
        total_reward = result['total_reward'].sum()
        wins = len(result[result['total_reward'] > 0])
        losses = len(result[result['total_reward'] < 0])

        return total_reward, wins, losses
    

    for timeframe in TIMEFRAMES:
        try:
            # Get Train and Test data for timeframe
            train, test, df = get_data(timeframe)
            # Process timeframe
            base.logger2.info(f" vb_genericAlgo_strategy: {ticker} / {timeframe} ".center(60, "*"))
        except KeyError as e: # 히스토리 레코드가 1건이라 볼린저밴드 20 을 만들수 없음.
            base.logger.error(f"vb_genericAlgo_strategy Key Error ({ticker} / {timeframe}): {e}")
            base.logger2.error(f"vb_genericAlgo_strategy Key Error ({ticker} / {timeframe}): {e}")
            continue
        except Exception as e:
            base.logger.error(f"vb_genericAlgo_strategy Error ({ticker} / {timeframe}): {e}")
            base.logger2.error(f"vb_genericAlgo_strategy Error ({ticker} / {timeframe}): {e}")

        with tqdm(total=GENERATIONS) as pbar:
            # Create Genetic Algorithm
            ga_instance = pygad.GA(num_generations=GENERATIONS,
                                num_parents_mating=5,
                                fitness_func=fitness_func,
                                sol_per_pop=POPULATIONS,
                                num_genes=3,
                                gene_space=[{'low': 0, 'high':1}, {'low': 0, 'high':1}, {'low': 0, 'high':1}],
                                parent_selection_type="sss",
                                crossover_type="single_point",
                                mutation_type="random",
                                mutation_num_genes=1,
                                keep_parents=-1,
                                on_generation=lambda _: pbar.update(1),
                                )
            # Run the Genetic Algorithm
            ga_instance.run()


        # logger2.info 정보가 너무 많아 TEST 결과 승률이 80% 이상인 경우만 display 하기 위하여 일부 display 순서 변경 20240122
        try:
            # Show details of the best solution.
            solution, solution_fitness, _ = ga_instance.best_solution()

            # Get Reward from test data
            profit, wins, losses = get_result(test, solution[0], solution[1], solution[2])

            win_rate = (wins/(wins + losses) if wins + losses > 0 else 0) * 100
            if win_rate >= 80 and profit > 1200000:
                # 최적 변수값 찾기
                base.logger2.info(f' Volatility & Bollinger Band with Generic Algorithm Strategy: {ticker} Best Solution Parameters for {timeframe} Timeframe '.center(60, '*'))      
                base.logger2.info(f"Min Volatility   : {solution[0]:6.4f}")
                base.logger2.info(f"Max Perc to Buy  : {solution[1]:6.4f}")
                base.logger2.info(f"Min Perc to Sell : {solution[2]:6.4f}")

                # Show the final result
                base.logger2.info(f'***** {ticker} Result for timeframe {timeframe} (TEST) ')
                base.logger2.info(f'* Profit / Loss (B&H)      : {(test["close"].iloc[-1] - test["close"].iloc[0]) * (CASH // test["close"].iloc[0]):.2f}')
                base.logger2.info(f"* Profit / Loss (Strategy) : {profit:.2f}")
                base.logger2.info(f"* Wins / Losses  : {wins} / {losses}")
                base.logger2.info(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")
                base.logger2.info("")

                # graph @@@ 값 검증 추가 필요: 20240128
                graph = df.copy().reset_index(drop=True)
                graph = graph.sort_values(by='date')
                graph['date'] = pd.to_datetime(graph['date'])

                '''
                sells, buys, buf, buf2 .... 좀 더 주의깊게 재검증이 필요함. 일단 기능상 충족으로 넘어감. 20240128
                '''
                sells = graph[graph['close_percentage'] > 95]   # Selling Point
                buys = graph[graph['close_percentage'] < 5]   # Buying Point

                # graph 에서 sells를 뺀 나머지 구하기
                buf = pd.merge(graph, sells, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                # graph 에서 buys를 뺀 나머지 구하기
                buf2 = pd.merge(graph, buys, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1) 
                # print(buf)

                plt.figure(figsize=(18, 6))
                plt.plot(graph['date'][-60:], graph['close'][-60:], label='주가', color='black')
                plt.plot(graph['date'][-60:], graph['high_limit'][-60:], label='상단 볼린저 밴드', linestyle='--', color='red')
                plt.plot(graph['date'][-60:], graph['low_limit'][-60:], label='하단 볼린저 밴드', linestyle='--', color='green')
                plt.scatter(buf['date'][-1:], buf['close'][-1:], color='red', label='Selling Point') # 특정 일자에 추가적인 점 플로팅
                plt.scatter(buf2['date'][-1:], buf2['close'][-1:], color='green', label='Buying Point') # 특정 일자에 추가적인 점 플로팅                
                # 그래프에 제목과 레이블 추가
                plt.title(f'Volatility({solution[0]:6.2f}) & BB with GA Strategy: Reward ({profit:.0f}), Wins/Losses ({wins:.0f}/{losses:.0f}), Win Rate ({win_rate:.2f}%)')
                plt.xlabel('날짜')
                plt.ylabel('가격')
                plt.grid()
                plt.legend()

                plt.savefig(base.reports_dir + f'/strg_v_bb_ga_{ticker}_{timeframe}.png')

                # Get Reward from train data
                profit, wins, losses = get_result(train, solution[0], solution[1], solution[2])
                base.logger2.info(f'***** {ticker} Result for timeframe {timeframe} (TRAIN) ')
                base.logger2.info(f'* Profit / Loss (B&H)      : {(train["close"].iloc[-1] - train["close"].iloc[0]) * (CASH // train["close"].iloc[0]):.2f}')
                base.logger2.info(f"* Profit / Loss (Strategy) : {profit:.2f}")
                base.logger2.info(f"* Wins / Losses  : {wins} / {losses}")
                base.logger2.info(f"* Win Rate       : {win_rate:.2f}%")

            else:
                pass
        except Exception as e:
            base.logger.error(' >>> Exception4: {}'.format(e))


'''
Volatility & Bollinger Band with Generic Algorithm Strategy 2
- 기존 버전1 대비 ga 의 최적변수를 볼린저밴드의 lenth 와 std 구간을 만들어 최적화하는 변수를 찾는 방법으로 적용
'''
def vb_genericAlgo_strategy2(ticker:str, TIMEFRAMES:list):
    # Constants
    CASH = 1_000_000
    POPULATIONS = 20
    GENERATIONS = 50

    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    # Loading data, and split in train and test datasets
    def get_data(timeframe):
      
        df = pd.read_csv(base.data_dir + f'/{ticker}_hist_{timeframe}.csv')
        if df.empty:
            base.logger.error(f'Read csv file {ticker} / {timeframe} is Empty')
            return None
        else:
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
            if (timeframe == '1min') or (timeframe == '1hour'):
                train, test = train_test_split(df, test_size=0.25, random_state=1104)
            else: #1day
                _date = (datetime.now() - timedelta(days=365)).date().strftime('%Y-%m-%d')
                train = df[df['date'] < _date]
                test = df[df['date'] >= _date]

        return train, test, df


    # Define fitness function to be used by the PyGAD instance
    def fitness_func(self, solution, sol_idx):

        try:
            # Get Reward from train data
            reward, _, _, _ = get_result(train, solution[0], solution[1], solution[2], solution[3])
        except:
            reward = 0
            pass

        # Return the solution reward
        return reward

    # Define a reward function
    def get_result(df, buy_length, buy_std, sell_length, sell_std, is_test=False):

        # Round to 2 digit to avoid the Bollinger bands function to generate weird field names
        buy_std = round(buy_std, 2)
        sell_std = round(sell_std, 2)

        # Generate suffixes for Bollinger bands fields
        buy_suffix = f'{int(buy_length)}_{buy_std}'
        sell_suffix = f'{int(sell_length)}_{sell_std}'

        # Generate a copy to avoid changing the original data
        df = df.copy().reset_index(drop=True)
        df = df.sort_values(by='date')

        # Calculate Bollinger bands based on parameters
        if not f'BBL_{buy_suffix}' in df.columns:  #@@@
            df.ta.bbands(close=df['close'], length=buy_length, std=buy_std, append=True)
        if not f'BBU_{sell_suffix}' in df.columns:  #@@@
            df.ta.bbands(close=df['close'], length=sell_length, std=sell_std, append=True)
        df = df.dropna()

        try:
            # Buy Signal
            df['signal'] = np.where(df['close'] < df[f'BBL_{buy_suffix}'], 1, 0)
            # Sell Signal
            df['signal'] = np.where(df['close'] > df[f'BBU_{sell_suffix}'], -1, df['signal'])
        except:  # 530107.KS 히스토리가 1 레코드 밖에 없음.
            df['signal'] = 0
            # base.logger.error(f'vb_genericAlgo_strategy2: {ticker} can not make Borlenger Band.')  # 너무 많이 반복되어서리.... ㅜㅜ

        # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
        result = df[df['signal'] != 0]
        result = result[result['signal'] != result['signal'].shift()]
        if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
        if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]

        # Calculate the reward & result / operation
        result['reward'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)
        result['wins'] = np.where(result['reward'] > 0, 1, 0)
        result['losses'] = np.where(result['reward'] < 0, 1, 0)

        # Generate window and filter windows without operations
        result_window = result.set_index('date').resample('3M').agg(
            {'close':'last','reward':'sum','wins':'sum','losses':'sum'}).reset_index()

        min_operations = 252 # 1 Year
        result_window = result_window[(result_window['wins'] + result_window['losses']) != 0]

        # Generate the result
        wins = result_window['wins'].mean() if len(result_window) > 0 else 0
        losses = result_window['losses'].mean() if len(result_window) > 0 else 0
        reward = result_window['reward'].mean() if (min_operations < (wins + losses)) or is_test else -min_operations + (wins + losses)
        pnl = result_window['reward'].sum()

        return reward, wins, losses, pnl


    for timeframe in TIMEFRAMES:
        # Get Train and Test data for timeframe
        train, test, df = get_data(timeframe)

        # Process data
        base.logger2.info(f" vb_genericAlgo_strategy2: {ticker} / {timeframe} ".center(60, "*"))

        with tqdm(total=GENERATIONS) as pbar:

            # Create Genetic Algorithm
            ga_instance = pygad.GA(num_generations=GENERATIONS,
                                num_parents_mating=5,
                                fitness_func=fitness_func,
                                sol_per_pop=POPULATIONS,
                                num_genes=4,
                                gene_space=[
                                    {'low': 1, 'high': 200, 'step': 1},
                                    {'low': 0.1, 'high': 3, 'step': 0.01},
                                    {'low': 1, 'high': 200, 'step': 1},
                                    {'low': 0.1, 'high': 3, 'step': 0.01}],
                                parent_selection_type="sss",
                                crossover_type="single_point",
                                mutation_type="random",
                                mutation_num_genes=1,
                                keep_parents=-1,
                                random_seed=42,
                                on_generation=lambda _: pbar.update(1),
                                )

            # Run the Genetic Algorithm
            ga_instance.run()

        # logger2.info 정보가 너무 많아 TEST 결과 승률이 80% 이상인 경우만 display 하기 위하여 일부 display 순서 변경 20240122
        # try:

        # Show details of the best solution.
        solution, solution_fitness, _ = ga_instance.best_solution()
        
        # Get result from test data
        # print(solution)
        reward, wins, losses, pnl = get_result(test, solution[0], solution[1], solution[2], solution[3], True)

        win_rate = (wins/(wins + losses) if wins + losses > 0 else 0) * 100
        if win_rate >= 80 and reward > 1200000:

            base.logger2.info(f'Volatility & Bollinger Band with Generic Algorithm Strategy 2: {ticker} Best Solution Parameters for {timeframe} Timeframe '.center(60, '*'))
            base.logger2.info('기존 버전1 대비 ga 의 최적변수를 볼린저밴드의 lenth 와 std 구간을 만들어 최적화하는 변수를 찾는 방법으로 적용')
            base.logger2.info(f'Buy Length    : {solution[0]:.0f}')
            base.logger2.info(f'Buy Std       : {solution[1]:.2f}')
            base.logger2.info(f'Sell Length   : {solution[2]:.0f}')
            base.logger2.info(f'Sell Std      : {solution[3]:.2f}')

            # Show the test result
            base.logger2.info(f'***** {ticker} Result for timeframe {timeframe} (TEST) ')
            base.logger2.info(f'* Reward                   : {reward:.2f}')
            base.logger2.info(f'* Profit / Loss (B&H)      : {(test["close"].iloc[-1] - test["close"].iloc[0]) * (CASH // test["close"].iloc[0]):.2f}')
            base.logger2.info(f'* Profit / Loss (Strategy) : {pnl:.2f}')
            base.logger2.info(f'* Wins / Losses            : {wins:.2f} / {losses:.2f}')
            base.logger2.info(f'* Win Rate                 : {win_rate:.2f}%')

            # graph @@@ 값 검증 추가 필요: 20240128
            graph = df.copy().reset_index(drop=True)
            graph = graph.sort_values(by='date')
            # print(int(solution[2]))
            # print(int(solution[0]))
            graph['Upper_MA'] = graph['close'].rolling(window=int(solution[2])).mean()
            # print(round(solution[3],2))
            # print(round(solution[1],2))
            # graph['Upper'] = graph['Upper_MA'] + 2 * round(solution[3],2)
            graph['Upper'] = graph['Upper_MA'] + 2 * round(graph['close'].rolling(window=int(solution[2])).std(),2)

            graph['Lower_MA'] = graph['close'].rolling(window=int(solution[0])).mean()

            # graph['Lower'] = graph['Lower_MA'] - 2 * round(solution[1],2)
            graph['Lower'] = graph['Lower_MA'] - 2 * round(graph['close'].rolling(window=int(solution[0])).std(),2)
            # print(graph['close'].rolling(window=int(solution[2])).std())
            # print(graph['close'].rolling(window=int(solution[0])).std())                

            # 주가가 밴드 상하단 사이의 몇 % 위치인지 확인
            graph['Position'] = (graph['close'] - graph['Lower']) / (graph['Upper'] - graph['Lower']) * 100

            '''
            sells, buys, buf, buf2 .... 좀 더 주의깊게 재검증이 필요함. 일단 기능상 충족으로 넘어감. 20240128
            '''
            sells = graph[graph['Position'] > 95]   # Selling Point
            buys = graph[graph['Position'] > 5]   # Buying Point

            # graph 에서 sells를 뺀 나머지 구하기
            buf = pd.merge(graph, sells, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
            # graph 에서 buys를 뺀 나머지 구하기
            buf2 = pd.merge(graph, buys, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1) 
            # print(buf)

            plt.figure(figsize=(18, 6))
            plt.plot(graph['date'][-60:], graph['close'][-60:], label='주가', color='black')
            # plt.plot(graph['date'], graph['Upper_MA'], label=f'{solution[2]:.2f}일 이동평균', linestyle='--', color='blue')
            # plt.plot(graph['date'], graph['Lower_MA'], label=f'{solution[0]:.2f}일 이동평균', linestyle='--', color='blue')                
            plt.plot(graph['date'][-60:], graph['Upper'][-60:], label='상단 볼린저 밴드', linestyle='--', color='red')
            plt.plot(graph['date'][-60:], graph['Lower'][-60:], label='하단 볼린저 밴드', linestyle='--', color='green')
            
            plt.scatter(buf['date'][-1:], buf['close'][-1:], color='red', label='Selling Point') # 특정 일자에 추가적인 점 플로팅
            plt.scatter(buf2['date'][-1:], buf2['close'][-1:], color='green', label='Buying Point') # 특정 일자에 추가적인 점 플로팅                
            # 그래프에 제목과 레이블 추가
            plt.title(f'Volatility & BB with GA Strategy2: Reward ({reward:.0f}), Wins/Losses ({wins:.0f}/{losses:.0f}), Win Rate ({win_rate:.2f}%)')
            plt.xlabel('날짜')
            plt.ylabel('가격')
            plt.grid()
            plt.legend()

            plt.savefig(base.reports_dir + f'/strg_v_bb_ga2_{ticker}_{timeframe}.png')

            # Get result from train data
            reward, wins, losses, pnl = get_result(train, solution[0], solution[1], solution[2], solution[3])

            # Show the train result
            base.logger2.info(f'***** {ticker} Result for timeframe {timeframe} (TRAIN) ')
            base.logger2.info(f'* Reward                   : {reward:.2f}')
            base.logger2.info(f'* Profit / Loss (B&H)      : {(train["close"].iloc[-1] - train["close"].iloc[0]) * (CASH // train["close"].iloc[0]):.2f}')
            base.logger2.info(f'* Profit / Loss (Strategy) : {pnl:.2f}')
            base.logger2.info(f'* Wins / Losses            : {wins:.2f} / {losses:.2f}')
            base.logger2.info(f'* Win Rate                 : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%')

        else:
            pass
        # except Exception as e:
        #     base.logger.error(' >>> Exception5: {}'.format(e))



'''
Generic Algorithm SellHoldBuy Strategy
- we will employ a genetic algorithm to update the network’s weights and biases.
- https://medium.com/@diegodegese/accelerating-model-training-and-improving-stock-market-predictions-with-genetic-algorithms-and-541b04be685b
'''
def gaSellHoldBuy_strategy(ticker):
    # Operations
    SELL = 0
    HOLD = 1
    BUY = 2
    # Constants
    OBS_SIZE = 32
    FEATURES = 2
    POPULATIONS = 20
    GENERATIONS = 50

    global model, observation_space_size, env

    class SellHoldBuyEnv(gym.Env):
        def __init__(self, observation_size, features, closes):
            # Data
            self.__features = features
            self.__prices = closes
            # Spaces
            self.observation_space = spaces.Box(low=np.NINF, high=np.PINF, shape=(observation_size,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)
            # Episode Management
            self.__start_tick = observation_size
            self.__end_tick = len(self.__prices)
            self.__current_tick = self.__end_tick
            # Position Management
            self.__current_action = HOLD
            self.__current_profit = 0
            self.__wins = 0
            self.__losses = 0
            
        def reset(self):
            # Reset the current action and current profit
            self.__current_action = HOLD
            self.__current_profit = 0
            self.__wins = 0
            self.__losses = 0            
            # Reset the current tick pointer and return a new observation
            self.__current_tick = self.__start_tick           
            return self.__get_observation()

        def step(self, action):
            # If current tick is over the last index in the feature array, the environment needs to be reset
            if self.__current_tick > self.__end_tick:
                raise Exception('The environment needs to be reset.')
            # Compute the step reward (Penalize the agent if it is stuck doing anything)
            step_reward = 0
            if self.__current_action == HOLD and action == BUY:
                self.__open_price = self.__prices[self.__current_tick]
                self.__current_action = BUY
            elif self.__current_action == BUY and action == SELL:            
                step_reward = self.__prices[self.__current_tick] - self.__open_price
                self.__current_profit += step_reward
                self.__current_action = HOLD               
                if step_reward > 0:
                    self.__wins += 1
                else:
                    self.__losses += 1
            # Generate the custom info array with the real and predicted values
            info = {
                'current_action': self.__current_action,
                'current_profit': self.__current_profit,
                'wins': self.__wins,
                'losses': self.__losses
            }
            # Increase the current tick pointer, check if the environment is fully processed, and get a new observation
            self.__current_tick += 1
            done = self.__current_tick >= self.__end_tick
            obs = self.__get_observation()
            # Returns the observation, the step reward, the status of the environment, and the custom information
            return obs, step_reward, done, info

        def __get_observation(self):
            # If current tick over the last value in the feature array, the environment needs to be reset
            if self.__current_tick >= self.__end_tick:
                return None
            # Generate a copy of the observation to avoid changing the original data
            obs = self.__features[(self.__current_tick - self.__start_tick):self.__current_tick]
            # Return the calculated observation
            return obs
    
    # Loading data, and split in train and test datasets   
    df = pd.read_csv(base.data_dir + f'/{ticker}_hist_1day.csv')

    df.ta.bbands(close=df['close'], length=20, append=True)
    df = df.dropna()
    pd.options.mode.chained_assignment = None
    try:
        df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
        df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
        df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
        df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1
    except KeyError as e: # 530107.KS 히스토리가 1 레코드 밖에 없음.
        base.logger.error(f"gaSellHoldBuy_strategy Key Error ({ticker}): {e}")
        base.logger2.error(f"gaSellHoldBuy_strategy Key Error ({ticker}): {e}")
        return None

    _date = (datetime.now() - timedelta(days=365)).date().strftime('%Y-%m-%d')
    train = df[df['date'] < _date]
    test = df[df['date'] >= _date]


    # Define fitness function to be used by the PyGAD instance
    def fitness_func(self, solution, sol_idx):
        
        # global model, observation_space_size, env
        
        # Set the weights to the model
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
        model.set_weights(weights=model_weights_matrix)

        # Run a prediction over the train data
        observation = env.reset()
        total_reward = 0

        done = False    
        while not done:
            try:
                state = np.reshape(observation, [1, observation_space_size])
                #q_values = model.predict(state, verbose=0)
                q_values = predict(state, model_weights_matrix)
                action = np.argmax(q_values[0])
                observation, reward, done, info = env.step(action)
                total_reward += reward                
            except:  # 히스토리 레코드가 1건이라 볼린저밴드 20 을 만들수 없음.
                # base.logger.error(f"reshape error: {ticker}") # too much... ㅜㅜ
                total_reward = 0
                break
        
        # try:
        #     # Print the reward and profit
        #     print(f"Solution {sol_idx:3d} - Total Reward: {total_reward:10.2f} - Profit: {info['current_profit']:10.3f}")
        #     if sol_idx == (POPULATIONS-1):
                # base.logger2.info(f" gaSellHoldBuy_strategy: {ticker} / {timeframe}".center(60, "*"))
        # except:
        #     pass
            
        # Return the solution reward
        return total_reward

    def predict(X, W):
        X      = X.reshape((X.shape[0],-1))           #Flatten
        X      = X @ W[0] + W[1]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[2] + W[3]                      #Dense
        X[X<0] = 0                                    #Relu
        X      = X @ W[4] + W[5]                      #Dense
        X      = np.exp(X)/np.exp(X).sum(1)[...,None] #Softmax
        return X
        
    # Create a train environmant
    env = SellHoldBuyEnv(observation_size=OBS_SIZE, features=train[['close_percentage','volatility']].values, closes=train['close'].values)
    observation_space_size = env.observation_space.shape[0] * FEATURES
    action_space_size = env.action_space.n

    # Create Model
    model = Sequential()
    model.add(Dense(16, input_shape=(observation_space_size,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(action_space_size, activation='linear'))
    model.summary()

    # Create Genetic Algorithm
    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=POPULATIONS)

    ga_instance = pygad.GA(num_generations=GENERATIONS,
                        num_parents_mating=5,
                        initial_population=keras_ga.population_weights,
                        fitness_func=fitness_func,
                        parent_selection_type="sss",
                        crossover_type="single_point",
                        mutation_type="random",
                        mutation_percent_genes=10,
                        keep_parents=-1)

    # Run the Genetic Algorithm
    ga_instance.run()

    # Show details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    # Create a test environmant
    env = SellHoldBuyEnv(observation_size=OBS_SIZE, features=test[['close_percentage','volatility']].values, closes=test['close'].values)

    # Set the weights of the best solution to the model
    best_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=best_weights_matrix)

    # Run a prediction over the test data
    observation = env.reset()
    total_reward = 0

    done = False    
    while not done:
        try:
            state = np.reshape(observation, [1, observation_space_size])
        except:  # 히스토리 레코드가 1건이라 볼린저밴드 20 을 만들수 없음.
            base.logger.error(f"reshape error: {ticker}")
            return
        #q_values = model.predict(state, verbose=0)
        q_values = predict(state, best_weights_matrix)
        action = np.argmax(q_values[0])
        observation, reward, done, info = env.step(action)
        total_reward += reward

    try:
        win_rate = (info['wins']/(info['wins'] + info['losses']) if info['wins'] + info['losses'] > 0 else 0) * 100
        if win_rate >= 80 and info['current_profit'] > 1200000:
            base.logger2.info(f'Generic Algorithm SellHoldBuy Strategy Result of {ticker}'.center(80, '*'))    
            base.logger2.info(f"Ticker & Timeframe: {ticker} & 1 Day")  # 1min, 1hour 는  장기투자시 적절하지 않아 제외
            base.logger2.info(f"Fitness value of the best solution = {solution_fitness}")
            base.logger2.info(f"Index of the best solution : {solution_idx}")
            # Show the final result
            base.logger2.info(f"* Profit/Loss: {info['current_profit']:6.3f}")
            base.logger2.info(f"* Wins: {info['wins']} - Losses: {info['losses']}")
            base.logger2.info(f"* Win Rate: {win_rate:6.2f}%")
        else:
            pass
    except Exception as e:
        base.logger.error(' >>> Exception6: {}'.format(e))


'''
Generic Algorithm & MACD Indicator Strategy
- https://medium.datadriveninvestor.com/my-approach-to-use-the-macd-indicator-in-the-market-part-2-3958aff26d0a
'''
# def GaMacd_strategy():
#     # Constants
#     DEBUG = 0
#     CASH = 1_000_000
#     POPULATIONS = 30
#     GENERATIONS = 50
#     FILE_TRAIN = '../data/spy.train.csv.gz'
#     FILE_TEST = '../data/spy.test.csv.gz'
#     TREND_LEN = 7
#     MIN_TRADES_PER_DAY = 1
#     MAX_TRADES_PER_DAY = 10

#     # Configuration
#     np.set_printoptions(suppress=True)
#     pd.options.mode.chained_assignment = None

#     # Loading data, and split in train and test datasets
#     def get_data():
#         train = pd.read_csv(FILE_TRAIN, compression='gzip')
#         train['date'] = pd.to_datetime(train['date'])
#         train.ta.ppo(close=train['close'], append=True)
#         train = train.dropna().reset_index(drop=True)

#         test = pd.read_csv(FILE_TEST, compression='gzip')
#         test['date'] = pd.to_datetime(test['date'])
#         test.ta.ppo(close=test['close'], append=True)
#         test = test.dropna().reset_index(drop=True)

#         train = train[train['date'] > (test['date'].max() - pd.Timedelta(365 * 10, 'D'))]

#         return train, test

#     # Define fitness function to be used by the PyGAD instance
#     def fitness_func(self, solution, sol_idx):
#         # Get Reward from train data
#         reward, wins, losses, pnl = get_result(train, train_dates,
#                                     solution[           :TREND_LEN*1],
#                                     solution[TREND_LEN*1:TREND_LEN*2],
#                                     solution[TREND_LEN*2:TREND_LEN*3],
#                                     solution[TREND_LEN*3:TREND_LEN*4])
#         if DEBUG:
#             print(f'\n{reward:10.2f}, {pnl:10.2f}, {wins:6.0f}, {losses:6.0f}, {solution[TREND_LEN*1:TREND_LEN*2]}, {solution[TREND_LEN*3:TREND_LEN*4]}', end='')

#         # Return the solution reward
#         return reward
    



