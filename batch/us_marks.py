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

from scipy import signal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gym import spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from math import sqrt, exp

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

gtta = {'VNQ':20, 'GLD':10, 'DBC':10, 'IEF':5, 'LQD':5, 'BNDX':5, 'TLT':5, \
        'EEM':10, 'VEA':10, 'DWAS':5, 'SEIM':5, 'DFSV':5, 'DFLV':5}

TIMEFRAMES = ['1min', '1hour', '1day']

def find_5days_ago():
    _day5 = datetime.now() - timedelta(days=5)
    return _day5

_day5_ago = find_5days_ago()
day5_ago = _day5_ago.date().strftime('%Y-%m-%d')


'''
01. Simulate Multi-Asset Baskets With Correlated Price Paths
- https://medium.com/codex/simulate-multi-asset-baskets-with-correlated-price-paths-using-python-472cbec4e379
'''
def multiAsset_basket():
    sp500 = fred.get_series('SP500', observation_start=from_date_MT)    
    nasdaq_comp = fred.get_series('NASDAQCOM', observation_start=from_date_MT)
    oil_wti = fred.get_series('DCOILWTICO', observation_start=from_date_MT)
    bond_10y = fred.get_series('DGS10', observation_start=from_date_MT)
    dollar_idx = fred.get_series('DTWEXBGS', observation_start=from_date_MT)

    sp500_norm = (sp500-sp500.min()) / (sp500.max()-sp500.min())
    nasdaq_comp_norm = (nasdaq_comp-nasdaq_comp.min()) / (nasdaq_comp.max()-nasdaq_comp.min())
    oil_wti_norm = (oil_wti-oil_wti.min()) / (oil_wti.max()-oil_wti.min())
    bond_10y_norm = (bond_10y-bond_10y.min()) / (bond_10y.max()-bond_10y.min())
    dollar_idx_norm = (dollar_idx-dollar_idx.min()) / (dollar_idx.max()-dollar_idx.min())

    # Manually input number of stocks
    NUMBER_OF_ASSETS = 5
    ASSET_TICKERS = ["sp500", "Nasdaq", "Oil", "Bond_10y", "Dollar"]
    Vol_sp500 = sp500_norm.std()
    Vol_Nasdaq = nasdaq_comp_norm.std()
    Vol_oil = oil_wti_norm.std()
    Vol_bond_10y = bond_10y_norm.std()
    Vol_dollar = dollar_idx_norm.std()

    VOLATILITY_ARRAY =[Vol_sp500, Vol_Nasdaq, Vol_oil, Vol_bond_10y, Vol_dollar]
    temp = pd.DataFrame()
    temp= pd.concat([temp, sp500, nasdaq_comp, oil_wti, bond_10y, dollar_idx], axis=1)
    temp.columns = ['sp500_norm', 'nasdaq_comp_norm', 'oil_wti_norm', 'bond_10y_norm', 'dollar_idx_norm']
    temp.fillna(method='ffill', inplace=True)
    COEF_MATRIX = temp.corr()

    # Perform Cholesky decomposition on coefficient matrix
    R = np.linalg.cholesky(COEF_MATRIX)
    # Compute transpose conjugate (only for validation)
    RT = R.T.conj()
    # Reconstruct coefficient matrix from factorization (only for validation)
    logger2.info("Multi-Asset Baskets With Correlated Price".center(60, '*'))
    logger2.info(": \n" + str(COEF_MATRIX))
    # logger2.info(": \n" + str(np.dot(R, RT)))

    T = 250                                   # Number of simulated days
    asset_price_array = np.full((NUMBER_OF_ASSETS,T), 100.0) # Stock price, first value is simulation input 
    volatility_array = VOLATILITY_ARRAY       # Volatility (annual, 0.01=1%)
    r = 0.001                                 # Risk-free rate (annual, 0.01=1%)
    dt = 1.0 / T

    # Plot simulated price paths
    retry_cnt = 5
    fig = plt.figure(figsize=(16,4*retry_cnt))

    for i in range(retry_cnt):

        for t in range(1, T):
            # Generate array of random standard normal draws
            random_array = np.random.standard_normal(NUMBER_OF_ASSETS)
            # Multiply R (from factorization) with random_array to obtain correlated epsilons
            epsilon_array = np.inner(random_array,R)
            # Sample price path per stock
            for n in range(NUMBER_OF_ASSETS):
                dt = 1 / T 
                S = asset_price_array[n,t-1]
                v = volatility_array[n]
                epsilon = epsilon_array[n]
                # Generate new stock price
                if n == 0:
                    asset_price_array[n,t] = S * exp((r - 0.5 * v**2) * dt + v * sqrt(dt) * epsilon)
                else:
                    asset_price_array[n,t] = temp.iloc[t,n]+100*1
                asset_price_array[n,t] = S * exp((r - 0.5 * v**2) * dt + v * sqrt(dt) * epsilon)

        ax = fig.add_subplot(retry_cnt, 1,  i+1)
        array_day_plot = [t for t in range(T)]
        for n in range(NUMBER_OF_ASSETS):
            ax.plot(array_day_plot, asset_price_array[n],\
                                label = '{}'.format(ASSET_TICKERS[n]))

        plt.grid()
        plt.xlabel('Day')
        plt.ylabel('Asset price')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(reports_dir + '/us_m0001.png')







'''
1. stocks
1.1 Timing Model & GTTA (Global Tactical Asset Allocation) Strategy
Asset Class Trend Following¶
http://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461
BUY RULE: Buy when monthly price > 10-month SMA.
SELL RULE: Sell and move to cash when monthly price < 10-month SMA.
GTAA consists of five global asset classes: US stocks, foreign stocks, bonds, real estate and commodities.
'''
# 새로운 포트폴리오 구성하는 방안으로 설정하면.
def sma_strategy(tickers:list, short_sma=20, long_sma=200):
    data = pd.DataFrame()
    for tick in tickers:
        #Download ticker price data from yfinance
        ticker = yf.Ticker(tick)
        buf = ticker.history(period='36mo') # test: 10mo, real: 36mo
        #Calculate 10 and 20 days moving averages
        sma20 = buf.ta.sma(short_sma, append=True)
        sma200 = buf.ta.sma(long_sma, append=True)
        buf.ta.rsi(close="Close", length=14, append=True)        
        #Create a column with buy and sell signals
        buf['Ticker'] = tick
        buf['Signal'] = 0.0
        buf['Signal'] = sma20 - sma200
        buf['Pivot'] = np.where((buf['Signal'].shift(1)*buf['Signal']) < 0, 1, 0)  # 1로 되는 일자부터 매수 또는 매도후 현금
        data = pd.concat([data, buf])
        
    return data


def timing_strategy(tickers, short_sma, long_sma):
    result = sma_strategy(tickers, short_sma, long_sma)
    buf = result[result['Pivot'] == 1].reset_index()
    # 날짜를 기준으로 최대 날짜의 인덱스를 찾기
    latest_indices = buf.groupby('Ticker')['Date'].idxmax()
    # 최대 날짜의 거래 내역을 발췌
    latest_records = buf.loc[latest_indices]
    # Change rate 비율만큼 Buy/Sell 실행할것, 초기 설정은 임계값 상승돌파하면 75% 추가매수, 하락돌파하면 75% 매도
    pivot_tickers = latest_records[latest_records['Date']  >= day5_ago]  # for test: '2023-05-16'
    pivot_tickers['Change_rate'] = np.where((pivot_tickers['Signal']) > 0, 1.75, 0.25)
    logger2.info(f'##### {long_sma}일 이동평균과 {short_sma}일 이동평균: Timing Strategy 에 따라 매도/매수 비중 조절할 것 !!! #####')
    logger2.info(pivot_tickers)
    # 검증용 백데이터 제공
    tick = pivot_tickers['Ticker']
    df = pd.DataFrame()
    for t in tick:
        buf = result[result['Ticker'] == t].tail(3)
        df = pd.concat([df, buf])
    logger2.debug(df) # 검증시 사용


'''
1.2 Maximum drawdown Strategy
'''
def daily_returns(prices):
    res = (prices/prices.shift(1) - 1.0)[1:]
    res.columns = ['return']
    return res

def cumulative_returns(returns):
    res = (returns + 1.0).cumprod()
    res.columns = ['cumulative return']
    return res

def max_drawdown(cum_returns):
    max_returns = np.fmax.accumulate(cum_returns)
    res = cum_returns / max_returns - 1
    res.columns = ['max drawdown']
    return res

def max_dd_strategy(tickers:list):
    threshold_value = -0.3
    plt.figure(figsize=(16,4*len(tickers)))
    for i, tick in enumerate(tickers):
        ticker = yf.Ticker(tick)
        prices = ticker.history(period='12y')['Close'] # 12: life cycle
        dret = daily_returns(prices)
        cret = cumulative_returns(dret)
        ddown = max_drawdown(cret)
        ddown[ddown.values < -0.3]

        plt.subplot(len(tickers), 1, i + 1)
        plt.grid()
        plt.bar(ddown.index, ddown, color='royalblue')
        plt.title(ticker)
        plt.axhline(y=threshold_value, color='red', linestyle='--', label='Threshold')
        plt.xlabel('Date')
        plt.ylabel('Draw Down %')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/us_m0100.png')



'''
1.3 Volatility-Bollinger Bands Strategy
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

    logger2.info(f'********** Volatility-Bollinger Bands Strategy: Result of {ticker} - {timeframe} Timeframe '.center(60, '*'))
    logger2.info(f'* Profit/Loss: {profit:.2f}')
    logger2.info(f"* Wins: {wins} - Losses: {losses}")
    try:
        logger2.info(f"* Win Rate: {100 * (wins/(wins + losses)):6.2f}%")
    except Exception as e:
        logger.error('Exception: {}'.format(e))

def volatility_bollinger_strategy(ticker:str, TIMEFRAMES:list):
    # Iterate over each timeframe, apply the strategy and show the result
    for timeframe in TIMEFRAMES:
        df = pd.read_csv(data_dir + f'/{ticker}_hist_{timeframe}.csv')
        # Add the signals to each row
        df['signal'] = get_vb_signals(df)
        df2 = df[df['ticker'] == ticker]
        # Get the result of the strategy
        show_vb_stategy_result(timeframe, df2)


'''
1.4 Reversal Strategy
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

    logger2.info(f'********** Reversal Strategy: Result of {ticker} for {timeframe} Timeframe '.center(60, '*'))
    logger2.info(f'* Profit/Loss: {profit:.2f}')
    logger2.info(f"* Wins: {wins} - Losses: {losses}")
    try:
        logger2.info(f"* Win Rate: {100 * (wins/(wins + losses)):6.2f}%")  # if wins + losses == 0
    except Exception as e:
        logger.error('Exception: {}'.format(e))

def reversal_strategy(ticker:str, TIMEFRAMES:list):
    # Iterate over each timeframe, apply the strategy and show the result
    for timeframe in TIMEFRAMES:
        df = pd.read_csv(data_dir + f'/{ticker}_hist_{timeframe}.csv')
        # Add the signals to each row
        df['signal'] = get_reversal_signals(df)
        df2 = df[df['ticker'] == ticker]
        # Get the result of the strategy
        show_reversal_stategy_result(timeframe, df2)


'''
1.5 Trend Following Strategy
Whether the market is experiencing a bull run or a bearish downturn, 
the goal is to hop on the trend early and stay on 
until there is a clear indication that the trend has reversed.
'''
def trend_following_strategy(ticker:str):
    # Constants
    CASH = 10000                 # Cash in account
    STOP_LOSS_PERC = -2.0        # Maximum allowed loss
    TRAILING_STOP = -1.0         # Value percentage for trailing_stop
    TRAILING_STOP_TRIGGER = 2.0  # Percentage to start using the trailing_stop to "protect" earnings
    GREEN_BARS_TO_OPEN = 4       # Green bars required to open a new position

    for timeframe in TIMEFRAMES:
        file_name = data_dir + f'/{ticker}_hist_{timeframe}.csv'
        df = pd.read_csv(file_name)   

        df['date'] = pd.to_datetime(df['date'])
        # Calculate consecutive bars in the same direction
        df['bar_count'] = ((df['open'] < df['close']) != (df['open'].shift() < df['close'].shift())).cumsum()
        df['bar_count'] = df.groupby(['bar_count'])['bar_count'].cumcount() + 1
        df['bar_count'] = df['bar_count'] * np.where(df['open'].values < df['close'].values,1,-1)

        # Variables Initialization
        cash = CASH
        shares = 0
        last_bar = None
        operation_last = 'SELL'
        ts_trigger = 0
        sl_price = 0
        operation_last_old = None

        reversed_df = df[::-1] # 시작일자부터 BUY/SELL 를 정해서 계산해 올라와야 맞을듯. 

        # Generate operations
        for index, row in reversed_df.iterrows():
            date = row['date']
            # If there is no operation
            if operation_last == 'SELL':
                if row['close'] == 0:
                    continue
                if last_bar is None:
                    last_bar = row
                    continue
                if row['bar_count'] >= GREEN_BARS_TO_OPEN:
                    operation_last = 'BUY'
                    open_price = row['close']
                    ts_trigger = open_price * (1 + (TRAILING_STOP_TRIGGER / 100))
                    sl_price = open_price * (1 + (STOP_LOSS_PERC / 100))
                    shares = int(cash // open_price)
                    cash -= shares * open_price
                else:
                    last_bar = None
                    continue        
            # If the last operation was a purchase
            elif operation_last == 'BUY':
                if row['close'] < sl_price:
                    operation_last = 'SELL'
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

            if timeframe == '1day':
                if (operation_last != operation_last_old) and (date >= pd.Timestamp('2023-01-01')):
                    logger2.info(f"{date}: {operation_last:<5}: {round(open_price, 2):8} - Cash: {round(cash, 2):8} - Shares: {shares:4} - CURR PRICE: {round(row['close'], 2):8} ({index}) - CURR POS: {round(shares * row['close'], 2)}")
                operation_last_old = operation_last
            
            last_bar = row

        if shares > 0:
            cash += shares * last_bar['close']
            shares = 0
            open_price = 0

        logger2.info(f'********** Trend Following Strategy: RESULT of {ticker} - {timeframe} Timeframe '.center(76, '*'))
        logger2.info(f"Cash after Trade: {round(cash, 2):8}")
        logger2.info('  ')






'''
1.6 ControlChartStrategy
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
    ticker_file = data_dir + f'/{ticker}.csv'
    default_window = 10
    CASH = 10_000
    DEFAULT_WINDOW = 10
    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    def get_data(ticker_file):
        df = pd.read_csv(ticker_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').resample('5T').agg('last')
        df = df.dropna()
        df['feature'] = signal.detrend(df['close'])
        return df.reset_index(drop=True)
    
    # Show result based on the selected rule
    def show_result(df, signal_field):
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
        # Show Result
        logger2.info(f' Result of {ticker} for ({signal_field}) '.center(60, '*'))
        logger2.info(f"* Profit / Loss  : {pnl:.2f}")
        logger2.info(f"* Wins / Losses  : {wins} / {losses}")
        logger2.info(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")
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

    df = get_data(ticker_file)

    logger2.info('         ')
    logger2.info('Rule 1 — One Point Beyond the 3σ Control Limit')
    logger2.info('Rule 2 — Eight or More Points on One Side of the Centerline Without Crossing')
    logger2.info('Rule 3 — Four out of five points in zone B or beyond')
    logger2.info('Rule 4 — Six Points or More in a Row Steadily Increasing or Decreasing')
    logger2.info('Rule 5 — Two out of three points in zone A')
    logger2.info('Rule 6 – 14 Points in a Row Alternating Up and Down')
    
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
1.7 Volatility & Bollinger Band with Generic Algorithm Strategy
'''
def vb_genericAlgo_strategy(ticker):
    # Constants
    POPULATIONS = 20
    GENERATIONS = 50
    CASH = 10_000

    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    # Loading data, and split in train and test datasets
    def get_data(timeframe):
        df = pd.read_csv(data_dir + f'/{ticker}_hist_{timeframe}.csv')
        df.ta.bbands(close=df['close'], length=20, append=True)
        df = df.dropna()
        df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
        df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
        df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
        df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1

        train, test = train_test_split(df, test_size=0.25, random_state=1104)
        # train = df[df['date'] < '2023-01-01']
        # test = df[df['date'] >= '2023-01-01']
        return train, test
    
    # Define fitness function to be used by the PyGAD instance
    def fitness_func(self, solution, sol_idx):
        # total reward 가 최대값을 갖을 수 있는 solution[0],[1],[2] 의 변수들을 찾아서 최적화(=> pygad.GA()를 통해서)
        total_reward, _, _ = get_result(train, solution[0], solution[1], solution[2])
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
        # Get Train and Test data for timeframe
        train, test = get_data(timeframe)
        # Process timeframe
        logger2.info("".center(60, "*"))

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

        # Show details of the best solution.
        solution, solution_fitness, _ = ga_instance.best_solution()

        logger2.info(f' Volatility & Bollinger Band with Generic Algorithm Strategy: {ticker} Best Solution Parameters for {timeframe} Timeframe '.center(60, '*'))      
        logger2.info(f"Min Volatility   : {solution[0]:6.4f}")
        logger2.info(f"Max Perc to Buy  : {solution[1]:6.4f}")
        logger2.info(f"Min Perc to Sell : {solution[2]:6.4f}")

        # Get Reward from train data
        profit, wins, losses = get_result(train, solution[0], solution[1], solution[2])
        logger2.info(f' {ticker} Result for timeframe {timeframe} (TRAIN) '.center(60, '*'))
        logger2.info(f'* Profit / Loss (B&H)      : {(train["close"].iloc[-1] - train["close"].iloc[0]) * (CASH // train["close"].iloc[0]):.2f}')
        logger2.info(f"* Profit / Loss (Strategy) : {profit:.2f}")
        logger2.info(f"* Wins / Losses  : {wins} / {losses}")
        logger2.info(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")

        # Get Reward from test data
        profit, wins, losses = get_result(test, solution[0], solution[1], solution[2])
        # Show the final result
        logger2.info(f' {ticker} Result for timeframe {timeframe} (TEST) '.center(60, '*'))
        logger2.info(f'* Profit / Loss (B&H)      : {(test["close"].iloc[-1] - test["close"].iloc[0]) * (CASH // test["close"].iloc[0]):.2f}')
        logger2.info(f"* Profit / Loss (Strategy) : {profit:.2f}")
        logger2.info(f"* Wins / Losses  : {wins} / {losses}")
        logger2.info(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")
        logger2.info("")



'''
1.8 Volatility & Bollinger Band with Generic Algorithm Strategy 2
- 기존 버전1 대비 ga 의 최적변수를 볼린저밴드의 lenth 와 std 구간을 만들어 최적화하는 변수를 찾는 방법으로 적용
'''
def vb_genericAlgo_strategy2(ticker):
    # Constants
    CASH = 10_000
    POPULATIONS = 30
    GENERATIONS = 50

    # Configuration
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None

    # Loading data, and split in train and test datasets
    def get_data(timeframe):
        df = pd.read_csv(data_dir + f'/{ticker}_hist_{timeframe}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna()
        train, test = train_test_split(df, test_size=0.25, random_state=1104)

        return train, test


    # Define fitness function to be used by the PyGAD instance
    def fitness_func(self, solution, sol_idx):

        # Get Reward from train data
        reward, _, _, _ = get_result(train, solution[0], solution[1], solution[2], solution[3])

        # Return the solution reward
        return reward

    # Define a reward function
    def get_result(df, buy_length, buy_std, sell_length, sell_std, is_test=False):

        # Round to 2 digit to avoid the Bollinger bands function to generate weird field names
        buy_std = round(buy_std, 3)
        sell_std = round(sell_std, 3)

        # Generate suffixes for Bollinger bands fields
        buy_suffix = f'{int(buy_length)}_{buy_std}'
        sell_suffix = f'{int(sell_length)}_{sell_std}'

        # Generate a copy to avoid changing the original data
        df = df.copy().reset_index(drop=True)

        # Calculate Bollinger bands based on parameters
        if not f'BBU_{buy_suffix}' in df.columns:
            df.ta.bbands(close=df['close'], length=buy_length, std=buy_std, append=True)
        if not f'BBU_{sell_suffix}' in df.columns:
            df.ta.bbands(close=df['close'], length=sell_length, std=sell_std, append=True)
        df = df.dropna()

        # Buy Signal
        df['signal'] = np.where(df['close'] < df[f'BBL_{buy_suffix}'], 1, 0)

        # Sell Signal
        df['signal'] = np.where(df['close'] > df[f'BBU_{sell_suffix}'], -1, df['signal'])

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
        # Get Train and Test data
        train, test = get_data(timeframe)

        # Process data
        logger2.info("".center(60, "*"))

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

        # Show details of the best solution.
        solution, solution_fitness, _ = ga_instance.best_solution()
        logger2.info(f'Volatility & Bollinger Band with Generic Algorithm Strategy 2: {ticker} Best Solution Parameters for {timeframe} Timeframe '.center(60, '*'))
        logger2.info('기존 버전1 대비 ga 의 최적변수를 볼린저밴드의 lenth 와 std 구간을 만들어 최적화하는 변수를 찾는 방법으로 적용')
        logger2.info(f'Buy Length    : {solution[0]:.0f}')
        logger2.info(f'Buy Std       : {solution[1]:.2f}')
        logger2.info(f'Sell Length   : {solution[2]:.0f}')
        logger2.info(f'Sell Std      : {solution[3]:.2f}')

        # Get result from train data
        reward, wins, losses, pnl = get_result(train, solution[0], solution[1], solution[2], solution[3])

        # Show the train result
        logger2.info(f' {ticker} Result for timeframe {timeframe} (TRAIN) '.center(60, '*'))
        logger2.info(f'* Reward                   : {reward:.2f}')
        logger2.info(f'* Profit / Loss (B&H)      : {(train["close"].iloc[-1] - train["close"].iloc[0]) * (CASH // train["close"].iloc[0]):.2f}')
        logger2.info(f'* Profit / Loss (Strategy) : {pnl:.2f}')
        logger2.info(f'* Wins / Losses            : {wins:.2f} / {losses:.2f}')
        logger2.info(f'* Win Rate                 : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%')

        # Get result from test data
        reward, wins, losses, pnl = get_result(test, solution[0], solution[1], solution[2], solution[3], True)

        # Show the test result
        logger2.info(f' {ticker} Result for timeframe {timeframe} (TEST) '.center(60, '*'))
        logger2.info(f'* Reward                   : {reward:.2f}')
        logger2.info(f'* Profit / Loss (B&H)      : {(test["close"].iloc[-1] - test["close"].iloc[0]) * (CASH // test["close"].iloc[0]):.2f}')
        logger2.info(f'* Profit / Loss (Strategy) : {pnl:.2f}')
        logger2.info(f'* Wins / Losses            : {wins:.2f} / {losses:.2f}')
        logger2.info(f'* Win Rate                 : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%')




'''
1.9 Generic Algorithm SellHoldBuy Strategy
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
    df = pd.read_csv(data_dir + f'/{ticker}_hist_1day.csv')
    df.ta.bbands(close=df['close'], length=20, append=True)
    df = df.dropna()
    pd.options.mode.chained_assignment = None
    df['high_limit'] = df['BBU_20_2.0'] + (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
    df['low_limit'] = df['BBL_20_2.0'] - (df['BBU_20_2.0'] - df['BBL_20_2.0']) / 2
    df['close_percentage'] = np.clip((df['close'] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
    df['volatility'] = df['BBU_20_2.0'] / df['BBL_20_2.0'] - 1
    train = df[df['date'] < '2023-01-01']
    test = df[df['date'] >= '2023-01-01']

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
            state = np.reshape(observation, [1, observation_space_size])
            #q_values = model.predict(state, verbose=0)
            q_values = predict(state, model_weights_matrix)
            action = np.argmax(q_values[0])
            observation, reward, done, info = env.step(action)
            total_reward += reward
        
        # Print the reward and profit: @@@
        # print(f"Solution {sol_idx:3d} - Total Reward: {total_reward:10.2f} - Profit: {info['current_profit']:10.3f}")
        # if sol_idx == (POPULATIONS-1):
        #     print(f"".center(60, "*"))
            
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

    logger2.info(f'Generic Algorithm SellHoldBuy Strategy Result of {ticker}'.center(80, '*'))    
    logger2.info(f"Ticker & Timeframe: {ticker} & 1 Day")  # 1min, 1hour 는  장기투자시 적절하지 않아 제외
    logger2.info(f"Fitness value of the best solution = {solution_fitness}")
    logger2.info(f"Index of the best solution : {solution_idx}")

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
        state = np.reshape(observation, [1, observation_space_size])
        #q_values = model.predict(state, verbose=0)
        q_values = predict(state, best_weights_matrix)
        action = np.argmax(q_values[0])
        observation, reward, done, info = env.step(action)
        total_reward += reward

    # Show the final result
    logger2.info(f"* Profit/Loss: {info['current_profit']:6.3f}")
    logger2.info(f"* Wins: {info['wins']} - Losses: {info['losses']}")
    logger2.info(f"* Win Rate: {100 * (info['wins']/(info['wins'] + info['losses'])):6.2f}%")


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
        # Get Reward from train data
        reward, wins, losses, pnl = get_result(train, train_dates,
                                    solution[           :TREND_LEN*1],
                                    solution[TREND_LEN*1:TREND_LEN*2],
                                    solution[TREND_LEN*2:TREND_LEN*3],
                                    solution[TREND_LEN*3:TREND_LEN*4])
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
    multiAsset_basket()


    '''
    1. Stock
    '''
    timing_strategy(gtta.keys(), 20, 200) # 200일 이평 vs 20일 이평
    timing_strategy(gtta.keys(), 1, 200) # 200일 이평 vs 어제 종가
    max_dd_strategy(WATCH_TICKERS) # max draw down strategy : 바닥에서 분할 매수구간 찾기

    # settings.py 에서 get_stock_history with timeframe 파일 만들어 줌. 
    for ticker in WATCH_TICKERS:
        print(ticker)
        get_stock_history(ticker, TIMEFRAMES)
    
    for ticker in WATCH_TICKERS:
        volatility_bollinger_strategy(ticker, TIMEFRAMES) # 임계값 찾는 Generic Algorithm 보완했음.

    for ticker in WATCH_TICKERS:
        vb_genericAlgo_strategy(ticker) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm       

    for ticker in WATCH_TICKERS:
        vb_genericAlgo_strategy2(ticker) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm           

    for ticker in WATCH_TICKERS:
        reversal_strategy(ticker, TIMEFRAMES) 

    for ticker in WATCH_TICKERS:     
        trend_following_strategy(ticker)  # 단기 매매 아님. 중장기 매매 기법, 1day 데이터만으로 실행

    for ticker in WATCH_TICKERS:
        control_chart_strategy(ticker)
        
    for ticker in WATCH_TICKERS:
        gaSellHoldBuy_strategy(ticker)
        

    # 2. Bonds
    # get_yields()