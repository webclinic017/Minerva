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
from scipy.stats import norm

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

pd.set_option('display.max_columns', None)  # 모든 열 표시
pd.set_option('display.width', None)        # 열 너비 제한 없음

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

global SIGMA_75, SIGMA_85
SIGMA_75 = norm.ppf(0.75)  # 75%에 해당하는 Z-score
SIGMA_85 = norm.ppf(0.85)  # 85%에 해당하는 Z-score

# 주요 관찰 대상국 설정 ('VN' 제외, Calendar에 없음.)
NATIONS = ['CN', 'EU', 'JP', 'KR', 'US', 'SG', 'DE', 'BR', 'IN']
NATIONS2 = ['CHN', 'EUR', 'JPN', 'KOR', 'USA', 'SGP', 'DEU', 'GBR', 'IND']
COUNTRIES = {
    "US":[{"alpha3":"USA"}, {"name":"United States"}, {"macrovar":"https://macrovar.com/united-states/"},],
    "KR":[{"alpha3":"KOR"}, {"name":"Korea"}, {"macrovar":"https://macrovar.com/south-korea/"},],
    "EU":[{"alpha3":"EUR"}, {"name":""}, {"macrovar":"https://macrovar.com/europe/"},],
    "JP":[{"alpha3":"JPN"}, {"name":"Japan"}, {"macrovar":"https://macrovar.com/japan/"},],
    "CN":[{"alpha3":"CHN"}, {"name":"China (People's Republic of)"}, {"macrovar":"https://macrovar.com/china/"},],
    "DE":[{"alpha3":"DEU"}, {"name":"Germany"}, {"macrovar":"https://macrovar.com/germany/"},],
    "IN":[{"alpha3":"IND"}, {"name":"India"}, {"macrovar":"https://macrovar.com/india/"},],
    "SG":[{"alpha3":"SGP"}, {"name":"Singapore"}, {"macrovar":"https://macrovar.com/singapore/"},],
}



ASSETS = ['stock', 'bond', 'commodity', 'cash']
# 보유종목들

# MY_TICKERS = {
#     'US':[{'bond':['TLT','EDV','HYG','LQD','TMF','IEF','SHV','SHY']}, {'stock':['SPY','QQQ','BAC','ENB','O','SCHD','SOXL','VNQ']}, {'commodity':['GLD']}, {'currency':['UUP']}, ],
#     'KR':[{'bond':['148070.KS','302190.KS','385560.KS','451530.KS','451670.KS','453850.KS']}, {'stock':['530107.KS','252670.KS']}, {'commodity':[]}, {'currency':[]}, ],    
#     'EU':[{'bond':[]}, {'stock':[]}, {'commodity':[]}, {'currency':[]}, ],
#     'JP':[{'bond':['1482.T','1487.T','2620.T','2621.T']}, {'stock':['1325.T','1545.T','1547.T','1571.T','1573.T','1678.T','2034.T','2563.T']}, 
#           {'commodity':[]}, {'currency':[]}, ],
#     'CN':[{'bond':[]}, {'stock':['510330.SS','002304.SZ','002594.SZ']}, {'commodity':[]}, {'currency':[]}, ],
#     'DE':[{'bond':['DBXG.DE','IBB1.DE','IS04.DE','IUSU.DE','VGEB.DE','VUTY.AS']}, {'stock':['BAS.DE','BNP.PA','CS.PA','GLE.PA']}, 
#           {'commodity':[]}, {'currency':[]}, ],    
#     'IN':[{'bond':[]}, {'stock':['','']}, {'commodity':[]}, {'currency':[]}, ],
#     'SG':[{'bond':['A35.SI']}, {'stock':['AJBU.SI','C6L.SI','CLR.SI','D05.SI','M44U.SI','ME8U.SI','Y92.SI','Z74.SI']}, 
#           {'commodity':[]}, {'currency':[]}, ],
# }

# 관심종목들
WATCH_TICKERS = {
    'US':[{'stock':['SPY','QQQ','BAC','ENB','O','SCHD','SOXL','VNQ']}, {'bond':['TLT','EDV','HYG','LQD','TMF','IEF','SHV','SHY']},  
          {'commodity':['GLD']}, {'currency':['UUP']}, ],
    'KR':[{'stock':['^KS200', '008770.KS','530107.KS','252670.KS','088980.KS',]}, {'bond':['148070.KS','302190.KS','385560.KS','451530.KS','451670.KS','453850.KS']},
          {'commodity':[]}, {'currency':[]}, ],    
    'EU':[{'stock':['^STOXX50E']}, {'bond':[]}, {'commodity':[]}, {'currency':[]}, ],
    'JP':[{'stock':['^N225', '1325.T','1545.T','1547.T','1571.T','1573.T','1678.T','2034.T','2563.T']}, {'bond':['1482.T','1487.T','2620.T','2621.T']},  
          {'commodity':[]}, {'currency':[]}, ],
    'CN':[{'stock':['510330.SS','002304.SZ', '^HSI', '002594.SZ']}, {'bond':[]}, {'commodity':[]}, {'currency':[]}, ],
    'DE':[{'stock':['BAS.DE','BNP.PA','CS.PA','GLE.PA']}, {'bond':['DBXG.DE','IBB1.DE','IS04.DE','IUSU.DE','VGEB.DE','VUTY.AS']},  
          {'commodity':[]}, {'currency':[]}, ],    
    'IN':[{'stock':['^BSESN','']}, {'bond':[]}, {'commodity':[]}, {'currency':[]}, ],
    'SG':[{'stock':['^STI', 'AJBU.SI','C6L.SI','CLR.SI','D05.SI','M44U.SI','ME8U.SI','Y92.SI','Z74.SI']}, {'bond':['A35.SI']}, 
          {'commodity':[]}, {'currency':[]}, ],
}

# 테스트용
WATCH_TICKERS2 = {
    'US':[{'stock':['SPY',]}, {'bond':['TLT',]}, {'commodity':[]}, {'currency':[]}, ],
    # 'KR':[{'stock':['530107.KS', '008770.KS',]}, {'bond':['148070.KS',]}, {'commodity':[]}, {'currency':[]}, ],  
    'KR':[{'stock':['302190.KS', '^KS200',]}, {'bond':[]}, {'commodity':[]}, {'currency':[]}, ],      
    'DE':[{'stock':[]}, {'bond':['VUTY.AS',]},
          {'commodity':[]}, {'currency':[]}, ],
}

urls = {
    'CN':'https://macrovar.com/china/',
    'EU':'https://macrovar.com/europe/', 
    'JP':'https://macrovar.com/japan/', 
    'KR':'https://macrovar.com/south-korea/', 
    'US':'https://macrovar.com/united-states/', 
    'SG':'https://macrovar.com/singapore/', 
    'DE':'https://macrovar.com/germany/', 
    'BR':'https://macrovar.com/brazil/',
    'IN':'https://macrovar.com/india/',
    # 'VN':'https://macrovar.com/vietnam/',
}

RESEARCHERS = ['OECD', 'IMF', 'WorldBank',]

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

# 날짜 및 시간 문자열을 날짜로 변환하는 함수
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()

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
            logger.error(' >>> ' + buf)
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
    except Exception as e:
        logger.error(' >>> ' + response)
        print("################# financilamodeling.com api error: "+ e)    

    # calendar = calendar.iloc[::-1]
    
    return calendar


# settings.py에 있는 financial modeling 에서 stock hitory 가져와 csv 파일로 저장하기까지. 
def get_stock_history_by_fmp(ticker:str, periods:list):  # period: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
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
        
    return df

# financial modeling 에서  Global Stock Market Indices 데이터 가져오기
def get_stock_indices():
    url = f'https://financialmodelingprep.com/api/v3/quotes/index?apikey={fmp_key}' 
    try:       
        df = requests.get(url).json()
        df = pd.DataFrame(df, columns=['symbol', 'name', 'price', 'changesPercentage', 'change', 'dayLow',\
                                        'dayHigh', 'yearHigh', 'yearLow', 'marketCap','priceAvg50',\
                                        'priceAvg200', 'exchange', 'volume', 'avgVolume', 'open', 'previousClose',\
                                        'eps', 'pe', 'earningsAnnouncement', 'sharesOutstanding', 'timestamp'])
    except Exception as e:
        print('Exception: {}'.format(e))    

    return df

# financial modeling 에서 각 상품별 선물 베팅 데이터 COT(Commitment of Traders)를 기간설정으로 가져오기, 최근 날짜건으로만. 
def get_cot_analysis_by_dates():
    url = f'https://financialmodelingprep.com/api/v4/commitment_of_traders_report_analysis?from={from_date_ST2}&to={to_date2}&apikey={fmp_key}' 
    try:       
        df = requests.get(url).json()
        df = pd.DataFrame(df, columns=['symbol', 'date', 'sector', 'currentLongMarketSituation', 'currentShortMarketSituation',\
                                        'marketSituation', 'previousLongMarketSituation', 'previousShortMarketSituation',\
                                        'previousMarketSituation', 'netPostion', 'previousNetPosition', 'changeInNetPosition',\
                                        'marketSentiment', 'reversalTrend', 'name', 'exchange'])
    except Exception as e:
        print('Exception: {}'.format(e))    

    return df

# financial modeling 에서 각 상품별 선물 베팅  COT(Commitment of Traders) 보고서를 기간설정으로 가져오기, 최근 날짜건으로만. 
def get_cot_report_by_dates():
    url = f'https://financialmodelingprep.com/api/v4/commitment_of_traders_report?from={from_date_ST2}&to={to_date2}&apikey={fmp_key}' 
    try:       
        df = requests.get(url).json()
        df = pd.DataFrame(df, columns=['symbol', 'date', 'short_name', 'sector', 'market_and_exchange_names',\
                                       'as_of_date_in_form_yymmdd', 'cftc_contract_market_code', 'cftc_market_code',\
                                       'cftc_region_code', 'cftc_commodity_code', 'open_interest_all', 'noncomm_positions_long_all',\
                                       'noncomm_positions_short_all', 'noncomm_postions_spread_all', 'comm_positions_long_all',\
                                       'comm_positions_short_all', 'tot_rept_positions_long_all', 'tot_rept_positions_short_all',\
                                       'nonrept_positions_long_all', 'nonrept_positions_short_all', 'open_interest_old',\
                                       'noncomm_positions_long_old', 'noncomm_positions_short_old', 'noncomm_positions_spread_old',\
                                       'comm_positions_long_old', 'comm_positions_short_old', 'tot_rept_positions_long_old',\
                                       'tot_rept_positions_short_old', 'nonrept_positions_long_old', 'nonrept_positions_short_old',\
                                       'open_interest_other', 'noncomm_positions_long_other', 'noncomm_positions_short_other',\
                                       'noncomm_positions_spread_other', 'comm_positions_long_other', 'comm_positions_short_other',\
                                       'tot_rept_positions_long_other', 'tot_rept_positions_short_other', 'nonrept_positions_long_other',\
                                       'nonrept_positions_short_other', 'change_in_open_interest_all', 'change_in_noncomm_long_all',\
                                       'change_in_noncomm_short_all', 'change_in_noncomm_spead_all', 'change_in_comm_long_all',\
                                       'change_in_comm_short_all', 'change_in_tot_rept_long_all', 'change_in_tot_rept_short_all',\
                                       'change_in_nonrept_long_all', 'change_in_nonrept_short_all', 'pct_of_open_interest_all',\
                                       'pct_of_oi_noncomm_long_all', 'pct_of_oi_noncomm_short_all', 'pct_of_oi_noncomm_spread_all',\
                                       'pct_of_oi_comm_long_all', 'pct_of_oi_comm_short_all', 'pct_of_oi_tot_rept_long_all',\
                                       'pct_of_oi_tot_rept_short_all', 'pct_of_oi_nonrept_long_all', 'pct_of_oi_nonrept_short_all',\
                                       'pct_of_open_interest_ol', 'pct_of_oi_noncomm_long_ol', 'pct_of_oi_noncomm_short_ol',\
                                       'pct_of_oi_noncomm_spread_ol', 'pct_of_oi_comm_long_ol', 'pct_of_oi_comm_short_ol', 'pct_of_oi_tot_rept_long_ol',\
                                       'pct_of_oi_tot_rept_short_ol', 'pct_of_oi_nonrept_long_ol', 'pct_of_oi_nonrept_short_ol', 'pct_of_open_interest_other',\
                                       'pct_of_oi_noncomm_long_other', 'pct_of_oi_noncomm_short_other', 'pct_of_oi_noncomm_spread_other',\
                                       'pct_of_oi_comm_long_other', 'pct_of_oi_comm_short_other', 'pct_of_oi_tot_rept_long_other',\
                                       'pct_of_oi_tot_rept_short_other', 'pct_of_oi_nonrept_long_other', 'pct_of_oi_nonrept_short_other',\
                                       'traders_tot_all', 'traders_noncomm_long_all', 'traders_noncomm_short_all', 'traders_noncomm_spread_all',\
                                       'traders_comm_long_all', 'traders_comm_short_all', 'traders_tot_rept_long_all', 'traders_tot_rept_short_all',\
                                       'traders_tot_ol', 'traders_noncomm_long_ol', 'traders_noncomm_short_ol', 'traders_noncomm_spead_ol',\
                                       'traders_comm_long_ol', 'traders_comm_short_ol', 'traders_tot_rept_long_ol', 'traders_tot_rept_short_ol',\
                                       'traders_tot_other', 'traders_noncomm_long_other', 'traders_noncomm_short_other', 'traders_noncomm_spread_other',\
                                       'traders_comm_long_other', 'traders_comm_short_other', 'traders_tot_rept_long_other', 'traders_tot_rept_short_other',\
                                       'conc_gross_le_4_tdr_long_all', 'conc_gross_le_4_tdr_short_all', 'conc_gross_le_8_tdr_long_all',\
                                       'conc_gross_le_8_tdr_short_all', 'conc_net_le_4_tdr_long_all', 'conc_net_le_4_tdr_short_all',\
                                       'conc_net_le_8_tdr_long_all', 'conc_net_le_8_tdr_short_all', 'conc_gross_le_4_tdr_long_ol', 'conc_gross_le_4_tdr_short_ol',\
                                       'conc_gross_le_8_tdr_long_ol', 'conc_gross_le_8_tdr_short_ol', 'conc_net_le_4_tdr_long_ol',\
                                       'conc_net_le_4_tdr_short_ol', 'conc_net_le_8_tdr_long_ol', 'conc_net_le_8_tdr_short_ol',\
                                       'conc_gross_le_4_tdr_long_other', 'conc_gross_le_4_tdr_short_other', 'conc_gross_le_8_tdr_long_other',\
                                       'conc_gross_le_8_tdr_short_other', 'conc_net_le_4_tdr_long_other', 'conc_net_le_4_tdr_short_other',\
                                       'conc_net_le_8_tdr_long_other', 'conc_net_le_8_tdr_short_other', 'contract_units'])
    except Exception as e:
        print('Exception: {}'.format(e))    

    return df


# financial modeling 에서 각 상품별 선물 베팅  COT(Commitment of Traders) 보고서를 기간설정으로 가져오기, 최근 날짜건으로만. 
def get_cot_report_by_symbol(symbol):
    url = f'https://financialmodelingprep.com/api/v4/commitment_of_traders_report/{symbol}?apikey={fmp_key}' 
    try:       
        df = requests.get(url).json()
        df = pd.DataFrame(df, columns=['symbol', 'date', 'short_name', 'sector', 'market_and_exchange_names',\
                                       'as_of_date_in_form_yymmdd', 'cftc_contract_market_code', 'cftc_market_code',\
                                       'cftc_region_code', 'cftc_commodity_code', 'open_interest_all', 'noncomm_positions_long_all',\
                                       'noncomm_positions_short_all', 'noncomm_postions_spread_all', 'comm_positions_long_all',\
                                       'comm_positions_short_all', 'tot_rept_positions_long_all', 'tot_rept_positions_short_all',\
                                       'nonrept_positions_long_all', 'nonrept_positions_short_all', 'open_interest_old',\
                                       'noncomm_positions_long_old', 'noncomm_positions_short_old', 'noncomm_positions_spread_old',\
                                       'comm_positions_long_old', 'comm_positions_short_old', 'tot_rept_positions_long_old',\
                                       'tot_rept_positions_short_old', 'nonrept_positions_long_old', 'nonrept_positions_short_old',\
                                       'open_interest_other', 'noncomm_positions_long_other', 'noncomm_positions_short_other',\
                                       'noncomm_positions_spread_other', 'comm_positions_long_other', 'comm_positions_short_other',\
                                       'tot_rept_positions_long_other', 'tot_rept_positions_short_other', 'nonrept_positions_long_other',\
                                       'nonrept_positions_short_other', 'change_in_open_interest_all', 'change_in_noncomm_long_all',\
                                       'change_in_noncomm_short_all', 'change_in_noncomm_spead_all', 'change_in_comm_long_all',\
                                       'change_in_comm_short_all', 'change_in_tot_rept_long_all', 'change_in_tot_rept_short_all',\
                                       'change_in_nonrept_long_all', 'change_in_nonrept_short_all', 'pct_of_open_interest_all',\
                                       'pct_of_oi_noncomm_long_all', 'pct_of_oi_noncomm_short_all', 'pct_of_oi_noncomm_spread_all',\
                                       'pct_of_oi_comm_long_all', 'pct_of_oi_comm_short_all', 'pct_of_oi_tot_rept_long_all',\
                                       'pct_of_oi_tot_rept_short_all', 'pct_of_oi_nonrept_long_all', 'pct_of_oi_nonrept_short_all',\
                                       'pct_of_open_interest_ol', 'pct_of_oi_noncomm_long_ol', 'pct_of_oi_noncomm_short_ol',\
                                       'pct_of_oi_noncomm_spread_ol', 'pct_of_oi_comm_long_ol', 'pct_of_oi_comm_short_ol', 'pct_of_oi_tot_rept_long_ol',\
                                       'pct_of_oi_tot_rept_short_ol', 'pct_of_oi_nonrept_long_ol', 'pct_of_oi_nonrept_short_ol', 'pct_of_open_interest_other',\
                                       'pct_of_oi_noncomm_long_other', 'pct_of_oi_noncomm_short_other', 'pct_of_oi_noncomm_spread_other',\
                                       'pct_of_oi_comm_long_other', 'pct_of_oi_comm_short_other', 'pct_of_oi_tot_rept_long_other',\
                                       'pct_of_oi_tot_rept_short_other', 'pct_of_oi_nonrept_long_other', 'pct_of_oi_nonrept_short_other',\
                                       'traders_tot_all', 'traders_noncomm_long_all', 'traders_noncomm_short_all', 'traders_noncomm_spread_all',\
                                       'traders_comm_long_all', 'traders_comm_short_all', 'traders_tot_rept_long_all', 'traders_tot_rept_short_all',\
                                       'traders_tot_ol', 'traders_noncomm_long_ol', 'traders_noncomm_short_ol', 'traders_noncomm_spead_ol',\
                                       'traders_comm_long_ol', 'traders_comm_short_ol', 'traders_tot_rept_long_ol', 'traders_tot_rept_short_ol',\
                                       'traders_tot_other', 'traders_noncomm_long_other', 'traders_noncomm_short_other', 'traders_noncomm_spread_other',\
                                       'traders_comm_long_other', 'traders_comm_short_other', 'traders_tot_rept_long_other', 'traders_tot_rept_short_other',\
                                       'conc_gross_le_4_tdr_long_all', 'conc_gross_le_4_tdr_short_all', 'conc_gross_le_8_tdr_long_all',\
                                       'conc_gross_le_8_tdr_short_all', 'conc_net_le_4_tdr_long_all', 'conc_net_le_4_tdr_short_all',\
                                       'conc_net_le_8_tdr_long_all', 'conc_net_le_8_tdr_short_all', 'conc_gross_le_4_tdr_long_ol', 'conc_gross_le_4_tdr_short_ol',\
                                       'conc_gross_le_8_tdr_long_ol', 'conc_gross_le_8_tdr_short_ol', 'conc_net_le_4_tdr_long_ol',\
                                       'conc_net_le_4_tdr_short_ol', 'conc_net_le_8_tdr_long_ol', 'conc_net_le_8_tdr_short_ol',\
                                       'conc_gross_le_4_tdr_long_other', 'conc_gross_le_4_tdr_short_other', 'conc_gross_le_8_tdr_long_other',\
                                       'conc_gross_le_8_tdr_short_other', 'conc_net_le_4_tdr_long_other', 'conc_net_le_4_tdr_short_other',\
                                       'conc_net_le_8_tdr_long_other', 'conc_net_le_8_tdr_short_other', 'contract_units'])
    except Exception as e:
        print('Exception: {}'.format(e))    

    return df


'''
국가별 경제전망을 지표화한 함수
- +1 과 -1 사이에서의 값 반환 
'''
# - 0.6744시그마 ~ 1.0364시그마 사이에서 탄력적으로 운용하기 위함. 
#   . 75% 분포를 갖는 시그마: 0.6744897501960817
#   . 85% 분포를 갖는 시그마: 1.0364333894937898
# - BULL 장에서는 대역폭을 2시그마(+- 42.5%)로 넓혀서 매도/매수선을 확대하고,
# - BEAR 장에서는 대역폭을 1.5시그마(+- 35%)로 좁혀서 매도/매수선을 긴축적으로 운영하고자 함.
def get_trend(ticker):
    country = 0 # defualt: STAY
    market = 0  # defualt: STAY
    country_weight = 0.667 # 가중치
    market_weight = 0.333 # 가중치

    _gdp = f"SELECT * FROM Calendars WHERE event like 'GDP Growth Rate%' AND country = '{country_sign}' \
            ORDER BY date DESC LIMIT 1"
    _cpi = f"SELECT * FROM Calendars WHERE event like '%cpi%' AND country = '{country_sign}' \
            AND estimate is NOT NULL ORDER BY date DESC LIMIT 1"
    _m2 = f"SELECT * FROM Indicators WHERE Indicator like '%M2%' AND Country = '{country_sign}'\
            ORDER BY date DESC LIMIT 1"
    
    def read_cals():
            try:
                gdp = pd.read_sql_query(_gdp, conn)
                print(gdp)
                if gdp['actual'][0] > gdp['estimate'][0]:
                    _multi_1 = 1.5
                elif gdp['actual'][0] == gdp['estimate'][0]:
                    _multi_1 = 1
                else:
                    _multi_1 = 0.5
                _multi_2 = gdp['change'][0]
                _multi_gpd = (_multi_1 + _multi_2)
                print(_multi_gpd)
                
                cpi = pd.read_sql_query(_cpi, conn)
                print(cpi)
                if cpi['actual'][0] > cpi['estimate'][0]:
                    _multi_3 = 1.5
                elif cpi['actual'][0] == cpi['estimate'][0]:
                    _multi_3 = 1
                else:
                    _multi_3 = 0.5
                _multi_4 = cpi['change'][0]
                _multi_cpi = (_multi_3 + _multi_4)
                print(_multi_cpi)   
                
                _multi_cont_tot = _multi_gpd - _multi_cpi
                if _multi_cont_tot > 1.2:
                    country = 'BULL'
                elif _multi_cont_tot < 0.8:
                    country = 'BEAR'
                else:
                    country = 'STAY'
                print(country, _multi_cont_tot)
            except Exception as e:
                print('Exception: {}'.format(e))

    trend = read_cals(ticker)





    if country == 'BULL':
        value_1 = 1
    elif country == 'BEAR':
        value_1 = -1
    else:
        value_1 = 0
    
    if market == 'BULL':
        value_2 = 1
    elif market == 'BEAR':
        value_2 = -1
    else:
        value_2 = 0 
        
    return country_weight*value_1 + market_weight*value_2





