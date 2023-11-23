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
from scipy import signal


# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

gtta = {'VNQ':20, 'GLD':10, 'DBC':10, 'IEF':5, 'LQD':5, 'BNDX':5, 'TLT':5, \
        'EEM':10, 'VEA':10, 'DWAS':5, 'SEIM':5, 'DFSV':5, 'DFLV':5}

MY_TICKERS = ['SPY', 'QQQ'] # only stocks
WATCH_TICKERS = ['SPY', 'QQQ'] # 관심종목들
COT_TICKERS = ['SPY', 'QQQ', 'UUP', 'FXY', 'TLT', 'VIXY', 'BCI']
COT_SYMBOLS = ['ES', 'NQ', 'VI', 'DX', 'BA', 'J6', 'ZB', 'ZN', 'SQ', 'CL', 'NG', 'GC', ]
# S&P 500 E-Mini (ES), Nasdaq 100 E-Mini (NQ), S&P 500 VIX (VI), US Dollar Index (DX), Bitcoin Micro (BA), Japanese Yen (J6), 
# 30-Year T-Bond (ZB), 10-Year T-Note (ZN), 3-Month SOFR (SQ), Crude Oil (CL), Natural Gas (NG), Gold (GC)

TIMEFRAMES = ['1min', '1hour', '1day']

def find_5days_ago():
    _day5 = datetime.now() - timedelta(days=5)
    return _day5

_day5_ago = find_5days_ago()
day5_ago = _day5_ago.date().strftime('%Y-%m-%d')



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
    print(df)


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

    print(f'********** Volatility-Bollinger Bands Strategy: Result of {ticker} for timeframe {timeframe} '.center(60, '*'))
    print(f'* Profit/Loss: {profit:.2f}')
    print(f"* Wins: {wins} - Losses: {losses}")
    try:
        print(f"* Win Rate: {100 * (wins/(wins + losses)):6.2f}%")
    except Exception as e:
        print('Exception: {}'.format(e))

def volatility_bollinger_strategy(ticker:str, TIMEFRAMES:list):
    # Iterate over each timeframe, apply the strategy and show the result
    for timeframe in TIMEFRAMES:
        df = pd.read_csv(data_dir + f'/us_d0130_{timeframe}.csv')
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

    print(f'********** Reversal Strategy: Result of {ticker} for timeframe {timeframe} '.center(60, '*'))
    print(f'* Profit/Loss: {profit:.2f}')
    print(f"* Wins: {wins} - Losses: {losses}")
    try:
        print(f"* Win Rate: {100 * (wins/(wins + losses)):6.2f}%")  # if wins + losses == 0
    except Exception as e:
        print('Exception: {}'.format(e))

def reversal_strategy(ticker:str, TIMEFRAMES:list):
    # Iterate over each timeframe, apply the strategy and show the result
    for timeframe in TIMEFRAMES:
        df = pd.read_csv(data_dir + f'/us_d0130_{timeframe}.csv')
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
def trend_following_strategy(ticker:str, df):
    # Constants
    CASH = 10000                 # Cash in account
    STOP_LOSS_PERC = -2.0        # Maximum allowed loss
    TRAILING_STOP = -1.0         # Value percentage for trailing_stop
    TRAILING_STOP_TRIGGER = 2.0  # Percentage to start using the trailing_stop to "protect" earnings
    GREEN_BARS_TO_OPEN = 4       # Green bars required to open a new position

    df['date'] = pd.to_datetime(df['date'])
    # Calculate consecutive bars in the same direction
    df['bar_count'] = ((df['open'] < df['close']) != (df['open'].shift() < df['close'].shift())).cumsum()
    df['bar_count'] = df.groupby(['bar_count'])['bar_count'].cumcount() + 1
    df['bar_count'] = df['bar_count'] * np.where(df['open'].values < df['close'].values,1,-1)

    # Variables Initialization
    cash = CASH
    shares = 0
    last_bar = None
    operation_last = 'WAIT'
    ts_trigger = 0
    sl_price = 0

    reversed_df = df[::-1] # 시작일자부터 Long/WAIT 를 정해서 계산해 올라와야 맞을듯. 

    # Generate operations
    for index, row in reversed_df.iterrows():
        date = row['date']
        # If there is no operation
        if operation_last == 'WAIT':
            if row['close'] == 0:
                continue
            if last_bar is None:
                last_bar = row
                continue
            if row['bar_count'] >= GREEN_BARS_TO_OPEN:
                operation_last = 'LONG'
                open_price = row['close']
                ts_trigger = open_price * (1 + (TRAILING_STOP_TRIGGER / 100))
                sl_price = open_price * (1 + (STOP_LOSS_PERC / 100))
                shares = int(cash // open_price)
                cash -= shares * open_price
            else:
                last_bar = None
                continue        
        # If the last operation was a purchase
        elif operation_last == 'LONG':
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

        print(f"{date}: {operation_last:<5}: {round(open_price, 2):8} - Cash: {round(cash, 2):8} - Shares: {shares:4} - CURR PRICE: {round(row['close'], 2):8} ({index}) - CURR POS: {round(shares * row['close'], 2)}")
        last_bar = row

    if shares > 0:
        cash += shares * last_bar['close']
        shares = 0
        open_price = 0

    print(f'********** Trend Following Strategy: RESULT of {ticker} '.center(76, '*'))
    print(f"Cash after Trade: {round(cash, 2):8}")

'''
1.6 The Commitment of Traders (COT) Report
https://wire.insiderfinance.io/download-sentiment-data-for-financial-trading-with-python-b07a35752b57
1) Insight into Market Sentiment
2) Early Warning Signals
3) Confirmation of Technical Analysis
4) Risk Management
5) Long-Term Investment Insights
6) Data-Driven Trading
'''

def get_dataframe(url):
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}
    df = pd.read_csv(url, compression='zip', storage_options=hdr, low_memory=False)
    df = df[['Market_and_Exchange_Names',
         'Report_Date_as_YYYY-MM-DD',
         'Pct_of_OI_Dealer_Long_All',
         'Pct_of_OI_Dealer_Short_All',
         'Pct_of_OI_Lev_Money_Long_All',
         'Pct_of_OI_Lev_Money_Short_All']]
    df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])

    return df

def set_cot_file():
    BUNDLE_URL = 'https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip'
    YEAR_URL = 'https://www.cftc.gov/files/dea/history/fut_fin_txt_{}.zip'

    df = get_dataframe(BUNDLE_URL) # 다 만들어진 파일을 가져오는 함수
    df = df[df['Report_Date_as_YYYY-MM-DD'] < '2016-01-01'] # 2016년것은 년도별 zip 파일에서 가져오니까. 그 전 것만 가져오도록 함.
    to_year = int((to_date2)[:4])+1  # 24년도 파일명에 23년 데이타가 들어가는군.
    for year in range(2016, to_year):
        tmp_df = get_dataframe(YEAR_URL.format(year)) # 다 만들어진 파일을 가져오는 함수
        df = pd.concat([df, tmp_df])
    df = df.sort_values(['Market_and_Exchange_Names','Report_Date_as_YYYY-MM-DD']).reset_index(drop=True)
    df = df.drop_duplicates()
    df.to_csv(data_dir + f'/market_sentiment_data.csv', index=False)

def get_cot_data(senti_file, SYMBOLS_SD_TO_MERGE, SYMBOL_SD, ticker_file, ticker):
        # Read Sentiment Data
        df_sd = pd.read_csv(senti_file)
        # Merge Symbols If Exists A Symbol With Different Names
        if SYMBOLS_SD_TO_MERGE is not None or len(SYMBOLS_SD_TO_MERGE) > 0:
            for symbol_to_merge in SYMBOLS_SD_TO_MERGE:
                df_sd['Market_and_Exchange_Names'] = df_sd['Market_and_Exchange_Names'].str.replace(symbol_to_merge, SYMBOL_SD)
        # Sort By Report Date
        df_sd = df_sd.sort_values('Report_Date_as_YYYY-MM-DD')
        # Filter Required Symbol
        df_sd = df_sd[df_sd['Market_and_Exchange_Names'] == SYMBOL_SD]
        df_sd['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df_sd['Report_Date_as_YYYY-MM-DD'])
        # Remove Unneeded Columns And Rename The Rest
        df_sd = df_sd.rename(columns={'Report_Date_as_YYYY-MM-DD':'report_date'})
        df_sd = df_sd.drop('Market_and_Exchange_Names', axis=1)

        # # Read / Get & Save Market Data
        # if not os.path.exists(ticker_file):
        #     ticker = yf.Ticker(ticker)
        #     df = ticker.history(
        #         interval='1d',
        #         start=min(df_sd['report_date']),
        #         end=max(df_sd['report_date']))
        #     df = df.reset_index()
        #     df['Date'] = df['Date'].dt.date
        #     df = df[['Date','Close']]
        #     df.columns = ['date', 'close']
        #     if len(df) > 0: df.to_csv(ticker_file, index=False)
        # else:
        #     df = pd.read_csv(ticker_file)
        
        # 어제 만들어진 ticker file 은 오늘 다시 업데이트 되지 않을텐데... 이상함...
        # Read / Get & Save Market Data
        ticker = yf.Ticker(ticker)
        df = ticker.history(
            interval='1d',
            start=min(df_sd['report_date']),
            end=max(df_sd['report_date']))
        df = df.reset_index()
        df['Date'] = df['Date'].dt.date
        df = df[['Date','Close']]
        df.columns = ['date', 'close']
        if len(df) > 0: df.to_csv(ticker_file, index=False)
        df = pd.read_csv(ticker_file)

        df['date'] = pd.to_datetime(df['date'])
        # Merge Market Sentiment Data And Market Data
        tolerance = pd.Timedelta('7 day')
        df = pd.merge_asof(left=df_sd,right=df,left_on='report_date',right_on='date',direction='backward',tolerance=tolerance)
        # Clean Data And Rename Columns
        df = df.dropna()
        df.columns = ['report_date', 'dealer_long', 'dealer_short', 'lev_money_long', 'lev_money_short', 'quote_date', 'close']

        return df

def get_cot_result(df, field, bb_length, min_bandwidth, max_buy_pct, min_sell_pct, CASH):
    # Generate a copy to avoid changing the original data
    df = df.copy().reset_index(drop=True)
    # Calculate Bollinger Bands With The Specified Field
    df.ta.bbands(close=df[field], length=bb_length, append=True)
    df['high_limit'] = df[f'BBU_{bb_length}_2.0'] + (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0']) / 2
    df['low_limit'] = df[f'BBL_{bb_length}_2.0'] - (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0']) / 2
    df['close_percentage'] = np.clip((df[field] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
    df['bandwidth'] = np.clip(df[f'BBB_{bb_length}_2.0'] / 100, 0, 1)
    df = df.dropna()
    # Buy Signal
    df['signal'] = np.where((df['bandwidth'] > min_bandwidth) & (df['close_percentage'] < max_buy_pct), 1, 0)
    # Sell Signal
    df['signal'] = np.where((df['close_percentage'] > min_sell_pct), -1, df['signal'])
    # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
    result = df[df['signal'] != 0]
    result = result[result['signal'] != result['signal'].shift()]
    if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
    if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]
    # Calculate the reward / operation
    result['total_reward'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)
    # Generate the result
    total_reward = result['total_reward'].sum()
    wins = len(result[result['total_reward'] > 0])
    losses = len(result[result['total_reward'] < 0])

    return total_reward, wins, losses


def cot_report_bat(ticker):
    # Configuration
    print('Bolenger Band Length: 20')
    np.set_printoptions(suppress=True)
    pd.options.mode.chained_assignment = None
    # Constants
    SYMBOL_SD = 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE'
    SYMBOLS_SD_TO_MERGE = ['E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE']
    senti_file = data_dir + f'/market_sentiment_data.csv'
    ticker_file = data_dir + f'/{ticker}.csv'
    CASH = 10_000
    BB_LENGTH = 20
    MIN_BANDWIDTH = 0
    MAX_BUY_PCT = 0.25
    MIN_SELL_PCT = 0.75

    # Get Required Data
    df = get_cot_data(senti_file, SYMBOLS_SD_TO_MERGE, SYMBOL_SD, ticker_file, ticker)
    # Get Result Based Calculating the BB on Each Field to Check Which is the Most Accurate
    for field in ['dealer_long', 'dealer_short', 'lev_money_long', 'lev_money_short']:
        total_reward, wins, losses = get_cot_result(df, field, BB_LENGTH, MIN_BANDWIDTH, MAX_BUY_PCT, MIN_SELL_PCT, CASH)
        print(f' Result of {ticker} for (Field: {field}) '.center(60, '*'))
        print(f"* Profit / Loss  : {total_reward:.2f}")
        print(f"* Wins / Losses  : {wins} / {losses}")
        print(f"* Win Rate       : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")
        

# def cot_report_on(symbols):
#     # get_oct_by_symbol(COT_SYMBOLS)
#     continue


'''
1.7 ControlChartStrategy
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
def get_control_data(ticker_file):
    df = pd.read_csv(ticker_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('5T').agg('last')
    df = df.dropna()
    df['feature'] = signal.detrend(df['close'])

    return df.reset_index(drop=True)


# def control_chart_strategy(ticker):
#     # Constants
#     ticker_file
#     default_window = 10
#     CASH = 10_000
#     # Configuration
#     np.set_printoptions(suppress=True)
#     pd.options.mode.chained_assignment = None



'''
Main Fuction
'''

if __name__ == "__main__":

    # 1. Stocks
    timing_strategy(gtta.keys(), 20, 200) # 200일 이평 vs 20일 이평
    timing_strategy(gtta.keys(), 1, 200) # 200일 이평 vs 어제 종가
    max_dd_strategy(WATCH_TICKERS) # max draw down strategy : 바닥에서 분할 매수구간 찾기

    # 아래와 같은 형태의 루프는 티커별로 함수를 돌리고, 해당 함수 안에서 루프를 더 돌리는 2차원 루프를 위한 구조임.
    for ticker in WATCH_TICKERS:
        get_stock_history(ticker, TIMEFRAMES)
    
    for ticker in WATCH_TICKERS:
        volatility_bollinger_strategy(ticker, TIMEFRAMES) # 임계값 찾는 Generic Algorithm 보완했음.

    for ticker in WATCH_TICKERS:
        reversal_strategy(ticker, TIMEFRAMES) # 임계값 찾는 Generic Algorithm 보완했음.

    file_name = data_dir + f'/us_d0130_1day.csv'
    df = pd.read_csv(file_name)
    for ticker in WATCH_TICKERS:
        df2 = df[df['ticker'] == ticker]
        trend_following_strategy(ticker, df2)  # 단기 매매 아님. 중장기 매매 기법, 1day 데이터만으로 실행

    set_cot_file()
    for ticker in COT_TICKERS:
        cot_report_bat(ticker)

    # for symbol in COT_SYMBOLS:  # financialmodeling.com 에서 해당 API 에 대한 비용을 요구하고 있음.
    #     cot_report_on(symbol)   # 유로화후 적용 예정
        
    
    # for ticker in WATCH_TICKERS:
    #     control_chart_strategy(ticker)
        
    
        
    



    # 2. Bonds
    # get_yields()