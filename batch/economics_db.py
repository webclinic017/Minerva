'''
# Prgram 명: make Macro Economic Tables in Economics Database
# 목적: 매일 올라오는 주요 국가별 경제지표들을 테이블에 추가하는 배치작업용
# Author: Jeonmin kang
# Mail: jarvisNim@gmail.com
# History
# 2023/11/07  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)
from settings import *

import re
import requests
from bs4 import BeautifulSoup as bs
import yfinance as yf
import os


'''
공통 영역
'''
# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 주요 관찰 대상국 설정
nations = ['CN', 'EU', 'JP', 'KR', 'US', 'SG', 'DE', 'BR', 'IN', 'VN']
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
    'VN':'https://macrovar.com/vietnam/',
}
# 검색기간 설정
_to_date = pd.to_datetime(to_date)
term_days = relativedelta(weeks=1)  # 초기 작업시는 12주 로 하면 사이트 부하오류 발생 안해서 최적!
from_date = (_to_date - term_days).date()
to_date = _to_date.date()



'''
데이터 베이스와 거시 경제 테이블 생성
'''
# Connect DataBase
database = database_dir+'/'+'Economics.db'
conn, engine = create_connection(database)


# financemodeling.com 추출한 경제 캘린더 <=== 검증완료: 수기 작성과 어플리케이션 작성 모두 멀티 Primary key 구성 가능
def create_Calendars(conn):
    with conn:
        cur = conn.cursor()
        cur.execute('CREATE TABLE if not exists Calendars (\
                date	TEXT NOT NULL,\
                country	TEXT NOT NULL,\
                event	TEXT NOT NULL,\
                currency	TEXT,\
                previous	NUMERIC,\
                estimate	NUMERIC,\
                actual	NUMERIC,\
                change	NUMERIC,\
                impact	TEXT,\
                changePercentage	NUMERIC,\
                PRIMARY KEY(date,country,event))')
    return conn

# MacroVar.com 에서 추출한 Financial Markets <== 현재는 사이트의 정보 업데이트가 미흡하여 미활용, 주기적으로 확인필요
def create_Markets(conn):
    with conn:
        cur = conn.cursor()
        cur.execute('CREATE TABLE if not exists Markets (\
        Country TEXT NOT NULL,\
        Market TEXT NOT NULL,\
        Symbol TEXT NOT NULL,\
        Last_value NUMERIC NOT NULL, \
        Momentum NUMERIC, \
        Trend NUMERIC, \
        Oscillator NUMERIC, \
        RSI NUMERIC, \
        DOD NUMERIC, \
        WOW NUMERIC, \
        MOM NUMERIC, \
        YOY NUMERIC, \
        Date TEXT,\
        PRIMARY KEY (Country, Market, Symbol, Last_value))')
    return conn

# MacroVar.com 에서 추출한 Macroeconomic Indicators
def create_Indicators(conn):
    with conn:
        cur = conn.cursor()
        cur.execute('CREATE TABLE if not exists Indicators (\
        Country TEXT NOT NULL,\
        Indicator TEXT NOT NULL,\
        Date TEXT NOT NULL,\
        Symbol TEXT,\
        Actual NUMERIC,\
        Previous NUMERIC,\
        MOM NUMERIC,\
        YOY NUMERIC,\
        Trend TEXT,\
        Slope TEXT,\
        ZS5Y INTEGER,\
        PRIMARY KEY (Country, Indicator, Date))')

    return conn

# 테이블 데이터 read
def read_table(table_name):
    M_table = table_name
    M_query = f"SELECT * from {M_table}"
    try:
        buf = pd.read_sql_query(M_query, conn)
        # logger2.info(buf)
        return buf
    except Exception as e:
        logger.error('Exception: {}'.format(e))

# 테이블 데이터 insert
def write_table(table_name, data):

    count = 0
    if table_name == 'Calendars':
        data = data.reset_index(drop=False)
        data.dropna(subset=['date', 'country', 'event'], inplace=True) # ??? 추가 확인 필요     
    elif table_name == 'Markets':
        data.dropna(subset=['Country', 'Market', 'Symbol', 'Last_value'], inplace=True) 
    elif table_name == 'Indicators':
        data.dropna(subset=['Country', 'Indicator', 'Date'], inplace=True)
    else:
        logger.error('Exception: Table Name Not found.')

    for i, d in enumerate(range(len(data))):
        buf = data.iloc[i:i+1:]
        try:
            _cnt = buf.to_sql(table_name, con=engine, if_exists='append', chunksize=1000, index=False)
            count += _cnt
        except:
            continue
    logger2.info(f'{table_name} insert Count: ' + str({count}))



# macrovar.com 에서 markets 표 읽어오기
def get_markets(country, url, table_name):
    page = requests.get(url, allow_redirects=True)
    soup = bs(page.text, "html.parser")
    table = soup.find_all(class_="container--tabs mb-2")

    # 1. Markets
    markets = pd.read_html(str(table))[0]

    buf = pd.DataFrame()
    buf['Country'] = [country] * len(markets)
    buf['Market'] = markets.Market
    buf['Symbol'] = markets.Symbol
    buf['Last_value'] = markets.Last
    buf['Momentum'] = markets.Mom
    buf['Trend'] = markets.Trend
    buf['Oscillator'] = markets.Exh
    buf['RSI'] = markets.RSI
    buf['DOD'] = markets['1D%']
    buf['WOW'] = markets['1W%']    
    buf['MOM'] = markets['1M%']
    buf['YOY'] = markets['1Y%']    
    buf['Date'] = pd.to_datetime(to_date2).date()

    return buf

# macrovar.com 에서 indicator 표 읽어오기
def get_indicators(country, url, table_name):
    page = requests.get(url, allow_redirects=True)
    soup = bs(page.text, "html.parser")
    table = soup.find_all(class_="container--tabs mb-2")

    # Macroeconomic Indicators
    indicators = pd.read_html(str(table))[1]
    indicators['Country'] = [country] *  len(indicators)
    indicators['Date'] = pd.to_datetime(indicators['Update']).dt.date

    buf = pd.DataFrame()
    buf['Country'] = indicators.Country
    buf['Indicator'] = indicators.Indicator
    buf['Date'] = indicators.Date
    buf['Symbol'] = indicators.Symbol
    buf['Actual'] = indicators.Actual
    buf['Previous'] = indicators.Previous
    buf['MOM'] = indicators['M/M%']
    buf['YOY'] = indicators['Y/Y%']
    buf['Trend'] = indicators.Trend
    buf['Slope'] = indicators.Slope
    buf['ZS5Y'] = indicators.ZS5Y

    return buf



'''
1. Calendars 테이블 데이터 구성
'''
def make_calendars(from_date, to_date):
    table_name = 'Calendars'
    cals = pd.DataFrame()
    for i in range(30):  # 최초 구성시는 20? 이후 매일 3회 배치작업으로 구성하고 있으니 1바퀴만 돌면 괜찮을듯.
        buf = get_calendar(from_date=from_date, to_date=to_date)
        for i in range(len(nations)):
            buf2 = buf[buf['country'] == nations[i]]
            cals = pd.concat([cals, buf2], axis=0)
        to_date = pd.to_datetime(from_date)
        from_date = (to_date - term_days).date()
        to_date = to_date.date()
    logger2.info(f'##### 최근 1주일동안의 Calendars 표 ####')
    logger2.info(cals)
    
    write_table(table_name, cals)


'''
2. Markets 테이블 데이터 구성
'''
def make_markets(**kwargs):
    table_name = 'Markets'
    df = pd.DataFrame()
    for key, value in kwargs.items():
        buf = get_markets(key, value, table_name)
        logger2.info(f'##### {buf.Country[0]} Markets 표 ####')
        logger2.info(buf)        
        df = pd.concat([df, buf])

    write_table(table_name, df)

'''
3. Indicators 테이블 데이터 구성
'''
def make_indicators(**kwargs):
    table_name = 'Indicators'
    df = pd.DataFrame()
    for key, value in kwargs.items():
        buf = get_indicators(key, value, table_name)
        logger2.info(f'##### {buf.Country[0]} Indicators 표 ####')
        logger2.info(buf)          
        df = pd.concat([df, buf])

    write_table(table_name, df)


'''
Main Fuction
'''

if __name__ == "__main__":

    # 테이블 생성 (최초 생성시)
    # create_Calendars(conn)
    # create_Markets(conn)
    # create_Indicators(conn)

    make_calendars(from_date, to_date)
    make_markets(**urls)
    make_indicators(**urls)






