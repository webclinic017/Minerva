# from inspect import CORO_RUNNING

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import investpy as inv # 20231123 삭제예정
import warnings
import time
import requests

import sqlite3
from time import sleep
from fredapi import Fred
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlite3 import Error
from dateutil.relativedelta import relativedelta
from tabulate import tabulate

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


'''
0. 공통영역 설정
'''
import logging, logging.config, logging.handlers


# 로그파일 텍스트의 길이 조정
class MaxLengthFilter(logging.Filter):
    def __init__(self, max_length):
        self.max_length = max_length

    def filter(self, record):
        record.msg = record.msg[:self.max_length]
        return True
## Loads The Config File
logging.config.fileConfig(batch_dir+'/logging.conf', disable_existing_loggers=False)

# 최대 길이 필터 설정 (예: 50자)
max_length = 3000
max_length_filter = MaxLengthFilter(max_length)
logging.getLogger().addFilter(max_length_filter)

# create a logger with the name from the config file. 
# This logger now has StreamHandler with DEBUG Level and the specified format in the logging.conf file
logger = logging.getLogger('batch')
logger2 = logging.getLogger('report')



fred = Fred(api_key='0e836827495d195023016a96b5fe6e4a')
bok_key = 'OLSJAN6H7R43WEYUEV5Q'
fmp_key = 'f57bdcaa7d140c9de35806d47fbd2f91'

warnings.filterwarnings('ignore')

now = datetime.today()
global to_date, to_date2, to_date3
to_date = now.strftime('%d/%m/%Y')
to_date2 = now.strftime('%Y-%m-%d')
to_date3 = now.strftime('%Y%m%d')
print('to_date: ', to_date)
print('to_date2: ', to_date2)
print('to_date3: ', to_date3)

global from_date_LT, from_date_MT, from_date_ST, from_date_LT2, from_date_MT2, from_date_ST2, from_date_LT3, from_date_MT3, from_date_ST3
# Used to analyze during 3 months for short term
_date = now + relativedelta(months=-3)
from_date_ST = _date.strftime('%d/%m/%Y')
from_date_ST2 = _date.strftime('%Y-%m-%d')
from_date_ST3 = _date.strftime('%Y%m%d')

# Used to analyze during 5 years for middle term (half of 10year Economic cycle)
_date = now + relativedelta(years=-5)
from_date_MT = _date.strftime('%d/%m/%Y')
from_date_MT2 = _date.strftime('%Y-%m-%d')
from_date_MT3 = _date.strftime('%Y%m%d')

# Used to analyze during 50 years for long term (5times of 10year Economic cycle)
_date = now + relativedelta(years=-50)
from_date_LT = _date.strftime('%d/%m/%Y') 
from_date_LT2 = _date.strftime('%Y-%m-%d')
from_date_LT3 = _date.strftime('%Y%m%d')

print('Short: ' + from_date_ST + '   Middle: ' + from_date_MT + '    Long: ' + from_date_LT)
# print('Short: ' + from_date_ST2 + '   Middle: ' + from_date_MT2 + '    Long: ' + from_date_LT2)
# print('Short: ' + from_date_ST3 + '   Middle: ' + from_date_MT3 + '    Long: ' + from_date_LT3)


'''
훗날 전체 경기전망과 전략의 점수를 산정하고 총 합으로 종합분석을 하고자 할때 사용하려고 만들어둠.

global IMPACT, STOCK_SIGNAL, BOND_SIGNAL, CASH_SIGNAL
# 중요도에 따라 1,2,3 부여
# 3등급은 대륙단위의 이벤트 대상: 전쟁, 트렌드,
# 2등급은 국가단위의 거시경제 지표들로 구성: 기준금리, M2,
# 1등급은 기업단위의 미시경제 지표들로 구성: 
IMPACT = 0
# 매 체크루틴에서 중요도에 따라 -5점부터 +5점까지중 2점단위로 부여
# 강력매도 -5, 매도 -3, 하락징후 -1, 상승징후 1, 매수 3, 강력매수 5
STOCK_SIGNAL = 0
BOND_SIGNAL = 0
CMDT_SIGNAL = 0  # Commodity
CASH_SIGNAL = 0
'''


# 신뢰구간
global CONF_INTVL
# CONF_INTVL = 3 # Critical Crack, 정규분포 이상치값을 매우매우 엄격하게 적용하는 경우
CONF_INTVL = 2 # Warning: 정규분포 이상치값을 엄격하게 적용하는 경우

# 상관계수(유사도)
global CONST_CORR
CONST_CORR = 0.85



Major_ETFs = [
    'AMLP', 'DUST', 'NUGT', 'JDST', 'XLE', 'XLF', 'QQQ', 'FXI', 'EWZ', 'EFA',
    'EEM', 'EZU', 'EWJ', 'IWM', 'ITB',  'UCO', 'UVXY', 'SQQQ', 'SPY', 'XHB', 
    'XOP', 'KRE', 'XLK', 'XLU', 'GDX', 'GDXJ', 'RSX', 'VEA', 'DGAZF', 'UWT', 
    'UGAZF', 'TVIX', 'HEDJ', 'EPI', 'DXJ', 'DBEF'
]

Most_ETFs = {
    'TQQQ':3, 'SQQQ':-3, 'SPY':1, 'SOXL':3, 'LUXY':-1, 'QQQ':1, 'XLF':1, 'SH':-1, 'LABU':3, 'EEM':1, 
    'PSQ':-1, 'EWZ':1, 'SPXU':-3, 'IWM':1, 'ARKK':1, 'FXI':1, 'XLE':1, 'GDX':1, 'HYG':1, 'SLV':1, 
    'LQD':1, 'SOXS':-3, 'LABD':-3, 'KWEB':1, 'EFA':1, 'SPXS':-3, 'TLT':1, 'XLP':1, 'TZA':-3, 'XBI':1, 
    'FNGU':3, 'VWO':1, 'KOLD':-2, 'JEPI':1, 'BKLN':1, 'GOVT':1, 'IEMG':1, 'XLU':1, 'VEA':1, 'XLI':1, 
    'XLV':1, 'FXI':1, 'EMB':1, 'VIXY':-1, 'SPXL':3, 'TNA':3, 'QID':-2, 'HYLB':1, 'SDS':-2, 'IEFA':1, 
    'EWU':1, 'SCO':-2, 'UPRO':2, 'VCIT':1, 'JNK':1, 'GDXJ':1
}


Most_Funds = [
    'Vanguard Total Stock Market Index Fund Institutional Plus Shares', 'Vanguard 500 Index Fund Admiral Shares',
    'Fidelity® 500 Index Fund', 'Vanguard Total Stock Market Index Fund Admiral Shares',
    'Fidelity® Government Money Market Fund', 'Fidelity® Government Cash Reserves',
    'Vanguard Total International Stock Index Fund Investor Shares', 'Vanguard Institutional Index Fund Institutional Plus Shares',
    'Fidelity® Contrafund® Fund', 'Fidelity® Contrafund® Fund Class K',
    'Vanguard Total Bond Market Ii Index Fund Investor Shares', 'Vanguard Total Bond Market Ii Index Fund Institutional Shares',
    'Vanguard Instl 500 Index Trust', 'American Funds The Growth Fund Of America® Class A',
    'Vanguard Institutional Index Fund Institutional Shares', 'Vanguard Total Bond Market Index Fund Admiral Shares',
    'Vanguard Wellington™ Fund Admiral™ Shares', 'American Funds American Balanced Fund® Class A',
    'Va Collegeamerica 529 Amcap 529a', 'Vanguard Cash Reserves Federal Money Market Fund Admiral Shares',
    'Dodge & Cox Stock Fund Class I', 'Vanguard Target Retirement 2030 Fund',
    'Vanguard Target Retirement 2035 Fund', 'Vanguard Target Retirement 2025 Fund',
    'American Funds The Income Fund Of America® Class A', 'American Funds Washington Mutual Investors Fund Class A',
    'Vanguard Total Stock Market Index Fund Institutional Shares', 'Vanguard Intermediate-term Tax-exempt Fund Admiral Shares',
    'Pimco Income Fund Institutional Class', 'American Funds Europacific Growth Fund® Class R-6'
]

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:

        engine = create_engine('sqlite:///'+db_file)
        conn = engine.connect()

        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn, engine


def include(filename):
    if os.path.exists(filename): 
        execfile(filename)


# Convert M(Million * 6), B(Billion * 9)
def convert_to_float(x, type=0):
    # type 0: only numeric,  type 1: object with comma (for excel, csv file)
    if type == 0:
        if 'K' in x:
            return float(x.strip('K'))*1000
        elif 'M' in x:
            return float(x.strip('M'))*1000000
        elif 'B' in x:
            return float(x.strip('B'))*1000000000
        else:
            return float(x)
    else:
        if 'K' in x:
            return f"{(float(x.strip('K'))*1000):,.2f}"
        elif 'M' in x:
            return f"{(float(x.strip('M'))*1000000):,.2f}"
        elif 'B' in x:
            return f"{(float(x.strip('B'))*1000000000):,.2f}"
        else:
            return float(x)

# 트렌드 디텍터
def trend_detector(data, col, tp_date_from, tp_date_to=to_date2, order=1):
    buf = np.polyfit(np.arange(len(data[tp_date_from:tp_date_to].index)), data[col][tp_date_from:tp_date_to], order)
    slope = buf[-2]
    
    return float(slope)

# Tipping Point 인자 추자: 20220913
def trend_detector_for_series(df, tp_date_from, tp_date_to=to_date2, order=1):
    data = df[df.index >= tp_date_from.strftime('%Y-%m-%d')]
    buf = np.polyfit(np.arange(len(data[tp_date_from:tp_date_to].index)), data[tp_date_from:tp_date_to], order)
    slope = buf[-2]
    
    return float(slope)
# def trend_detector_for_series(df, tp, order=1):
#     data = df.index[tp:]
#     buf = np.polyfit(range(len(data.index)), data, order)
#     slope = buf[-2]
#     return float(slope)

 # 이전 2개가 계속 커지고, 이후 2개가 계속 작아지는 구간이 Turning Point 로 찾아내는 기능
def turning_point(data, col):
    # nan 은 dropna() 로 제거
    result = []
    result = ((data[col].shift(-2) < data[col].shift(-1)) & (data[col].shift(-1) < data[col]) & \
    (data[col].shift(1) < data[col]) & (data[col].shift(2) < data[col].shift(1))).astype(int)
    result = result[result > 0]
    
    return result


def turning_point_for_series(data):
    result = []
    result = ((data.shift(-2) < data.shift(-1)) & (data.shift(-1) < data) & \
    (data.shift(1) < data) & (data.shift(2) < data.shift(1))).astype(int)
    result = result[result > 0]
              
    return result


def delete_Crack_By_Date(conn, table, date=to_date2):
    """
    Delete a Curr_Crack by Date (default=to_date)
    :param conn:  Connection to the SQLite database
    :param id: Date of the Curr_Crack (%Y-%m-%d)
    :return:
    """
    sql = f'DELETE FROM {table} WHERE Date=?'
    cur = conn.cursor()
    cur.execute(sql, (date,))
    conn.commit()   


def normalize(data):
    result = pd.DataFrame()
    result = (data-min(data))/(max(data)-min(data))
    
    return result



# 한국은행 통계시스템에서 데이터 가져오기 
def get_bok(bok_key,stat_code,cycle_type,start_date,end_date,item_1, item_2, item_3):
    import json
    from urllib.request import urlopen
    df = pd.DataFrame()
    file_type  = 'json'
    lang_type  = 'kr'
    start_no   = '1'
    end_no     = '5000'
    urls = []
    
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{bok_key}/{file_type}/{lang_type}/{start_no}/{end_no}/\
{stat_code}/{cycle_type}/{start_date}/{end_date}"
    
    for x in item_1:
        x = '/'+ x
        url += x
        for y in item_2:
            y = '/'+ y
            url += y
            for z in item_3:
                z = '/'+ z
                url += z
                urls.append(url)
                url = url.replace(z,'')
            if len(item_3) == 0:
                urls.append(url)
            url = url.replace(y,'')
        if len(item_2) == 0:
            urls.append(url)
        url = url.replace(x,'')
        # print(url)
    
    for x in urls:
        print(x)
        try:
            with urlopen(x) as response:
                html = response.read().decode('utf-8')
                data = json.loads(html)
                if data.get('RESULT') is None: # BOK API 는 200 성공이후 RESULT 값으로 input 값 검증결과 보내줌.
                    data = data['StatisticSearch']['row']
                    _ = pd.DataFrame(data)
                    df = pd.concat([df, _], axis=0)
                else:
                    raise AttributeError(data.get('RESULT'))
                sleep(1)
        except Exception as e:
            print(response.code,':',response.message)
            print(str(e))
    return df


def _get_ema(P, last_ema, N):
    return ((P - last_ema) * (2/(N+1)) + last_ema)


def get_ema(data:pd.DataFrame, N:int, key:str='Close'):
    data['SMA_' + str(N)] = data[key].rolling(N).mean()
    ema = np.zeros(len(data)) + np.nan
    for i, _row in enumerate(data.iterrows()):
        row = _row[1]
        if np.isnan(ema[i-1]):
            ema[i] =  row['SMA_' + str(N)]
        else:
            ema[i] = _get_ema(row[key], ema[i-1], N)
#         print(ema)
    data['EMA_' + str(N)] = ema.copy()
    return data


def get_macd(data:pd.DataFrame, N_fast:int, N_slow:int, signal_line:bool=True, N_sl:int=9):
    assert N_fast < N_slow, ("Fast EMA must be less than slow EMA parameter.")
    data = get_ema(data, N_fast)
    data = get_ema(data, N_slow)
    data['MACD'] = data[f'EMA_{N_fast}'] - data[f'EMA_{N_slow}']
    if signal_line:
        data = get_ema(data, N_sl, key='MACD')
        data.rename(
            columns={f'SMA_{N_sl}':f'SMA_MACD_{N_sl}', f'EMA_{N_sl}':f'SignalLine_{N_sl}'}, inplace=True
        )
    return data


def get_adx(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(lookback).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/lookback).mean()/atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/lookback).mean()/atr))
    dx = (abs(plus_di-minus_di) / abs(plus_di+minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
    
    return plus_di, minus_di, adx_smooth


# --------------------------------
# WILLIAMS %R CALCULATION
# --------------------------------
# W%R 14 = [ H.HIGH - C.PRICE ] / [ L.LOW - C.PRICE ] * ( - 100 )
# where,
# W%R 14 = 14-day Williams %R of the stock
# H.HIGH = 14-day Highest High of the stock
# L.LOW = 14-day Lowest Low of the stock
# C.PRICE = Closing price of the stock
def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr


# --------------------------------
# The True Strength Index (TSI)
# --------------------------------
# TSI LINE = [ DS. ACTUAL PC / DS. ABSOLUTE PC ] * 100
# where,
# DS. ACTUAL PC = Double smoothed actual price change with the length of 25 and 13
# DS. ABSOLUTE PC = Double smoothed absolute price change with the length of 25 and 13
# SIGNAL LINE = EXP.MA 13 [ TSI LINE ]
def get_tsi(close, long, short, signal):
    diff = close - close.shift(1)
    abs_diff = abs(diff)
    
    diff_smoothed = diff.ewm(span = long, adjust = False).mean()
    diff_double_smoothed = diff_smoothed.ewm(span = short, adjust = False).mean()
    abs_diff_smoothed = abs_diff.ewm(span = long, adjust = False).mean()
    abs_diff_double_smoothed = abs_diff_smoothed.ewm(span = short, adjust = False).mean()
    
    tsi = (diff_double_smoothed / abs_diff_double_smoothed) * 100
    signal = tsi.ewm(span = signal, adjust = False).mean()
    tsi = tsi[tsi.index >= '2020-01-01'].dropna()
    signal = signal[signal.index >= '2020-01-01'].dropna()
    
    return tsi, signal


# --------------------------------
# Disparity Index
# --------------------------------
# DI 14 = [ C.PRICE - MOVING  AVG 14 ] / [ MOVING AVG 14 ] * 100
# where,
# DI 14 = 14-day Disparity Index
# MOVING AVG 14 = 14-day Moving Average
# C.PRICE = Closing price of the stock
def get_di(data, lookback):
    ma = data.rolling(lookback).mean()
    di = ((data - ma) / ma) * 100
    return di



# --------------------------------
# Monitoring Multiple Feeds
# --------------------------------
# DI 14 = [ C.PRICE - MOVING  AVG 14 ] / [ MOVING AVG 14 ] * 100
# where,
# DI 14 = 14-day Disparity Index
# MOVING AVG 14 = 14-day Moving Average
# C.PRICE = Closing price of the stock
# def fetch_rss_data(url):
#     feed = feedparser.parse(url)
#     print("Feed Title:", feed.feed.title)
#     for entry in feed.entries:
#         print("Entry Title:", entry.title)
#         print("Entry Link:", entry.link)
#         print("Entry Published Date:", entry.published)
#         print("Entry Summary:", entry.summary)
#         print("\n")
# # List of RSS feed URLs
# rss_feed_urls = [
#     "https://www.example1.com/rss",
#     "https://www.example2.com/rss",
#     "https://www.example3.com/rss"
# ]
# # Fetch data from multiple RSS feeds
# for url in rss_feed_urls:
#     fetch_rss_data(url)


# plt.axvspan(datetime(1973,11,15), datetime(1975,3,15), facecolor='gray', edgecolor='gray', alpha=0.3)
# plt.axvspan(datetime(1980,2,15), datetime(1980,7,15), facecolor='gray', edgecolor='gray', alpha=0.3)
# plt.axvspan(datetime(1981,6,29), datetime(1982,11,1), facecolor='gray', edgecolor='gray', alpha=0.3)# Black Monday
# plt.axvspan(datetime(1990,6,2), datetime(1991,3,4), facecolor='gray', edgecolor='gray', alpha=0.3)
# plt.axvspan(datetime(2001,3,5), datetime(2001,11,5), facecolor='gray', edgecolor='gray', alpha=0.3)# Millaium Crisis
# plt.axvspan(datetime(2007,11,26), datetime(2009,6,1), facecolor='gray', edgecolor='gray', alpha=0.3)# Financial Crisis
# plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis



'''
작업결과 Database insert
'''
def db_insert(M_db, M_table, M_query, M_buffer, conn, engine, logger, logger2):
    M_buffer['Tot_Count'] =  M_buffer.iloc[:, 3:].sum(axis=1)
    M_buffer['Tot_Percent'] = M_buffer['Tot_Count']/(len(M_buffer.columns) - 3) * 100
    try:
        if M_db['Date'].str.contains(to_date2).any():
            buf = 'Duplicated: ' + M_db['Date']
            logger.error(buf)
            delete_Crack_By_Date(conn, 'Sent_Crack', date=to_date2)
        M_buffer.to_sql(M_table, con=engine, if_exists='append', chunksize=1000, index=False, method='multi')
    except Exception as e:
        print("################# Check Please: " + e)
    try:
        # display(pd.read_sql_query(M_query, conn)[-5:])
        buf = pd.read_sql_query(M_query, conn)[-5:]
        logger2.info(buf)
    except Exception as e:
        print('################# Exception: {}'.format(e))

    # 배치 프로그램 최종 종료시 Activate 후 실행
    conn.close()


'''
financialmodeling.com API 연계
US Economics & Markets 관련 데이터 수집해오는 API
'''

def get_daily_hist(ticker, from_date, to_date=to_date2):
    url = (f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={from_date}&to={to_date}&apikey={fmp_key}')
    response = requests.get(url).json()
    # Flatten data
    symbol = response.get('symbol')
    df = pd.json_normalize(response, record_path =['historical'])
    
    return symbol, df

def get_calendar(from_date, to_date=to_date2):
    url = (f'https://financialmodelingprep.com/api/v3/economic_calendar?from={from_date}&to={to_date}&apikey={fmp_key}')
    try:
        response = requests.get(url).json()
        calendar = pd.DataFrame(response)
        calendar = calendar.set_index('date')        
    except Exception as e:
        logger.error(response)
        print("################# financilamodeling.com api error: "+ e)    

    # calendar = calendar.iloc[::-1]
    
    return calendar


# settings.py에 있는 financial modeling 에서 stock hitory 가져와 csv 파일로 저장하기까지. 
def get_stock_history(ticker:str, periods:list):  # period: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
    for period in periods:
        url = f'https://financialmodelingprep.com/api/v3/historical-chart/{period}/{ticker}?from={from_date_MT2}&to={to_date2}&apikey={fmp_key}'
        try:
            buf = requests.get(url).json()
            df = pd.DataFrame(buf, columns=['date', 'open', 'low','high','close','volume'])
            df['ticker'] = ticker
            df.to_csv(data_dir + f'/{ticker}_hist_{period}.csv', index=False)
        except Exception as e:
            print('Exception: {}'.format(e))


# settings.py에 있는 financial modeling 에서 OCT Report 를 심볼로 가져오기
def get_oct_by_symbol(symbols:list):  # period: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
    for symbol in symbols:
        url = f'https://financialmodelingprep.com/api/v4/commitment_of_traders_report_analysis/{symbol}?apikey={fmp_key}'        
        try:
            buf = requests.get(url).json()
            df = pd.DataFrame(buf, columns=['symbol', 'date', 'sector', 'currentLongMarketSituation', \
                                            'currentShortMarketSituation', 'marketSituation',\
                                            'previousLongMarketSituation', 'previousShortMarketSituation', \
                                            'previousMarketSituation', 'netPostion','previousNetPosition', \
                                            'changeInNetPosition', 'marketSentiment', 'reversalTrend', 'name', 'exchange'])
        except Exception as e:
            print('Exception: {}'.format(e))