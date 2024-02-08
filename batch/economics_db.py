'''
# Prgram 명: make Macro Economic Tables in Economics Database
# 목적: 매일 올라오는 주요 국가별 경제지표들을 테이블에 추가하는 배치작업용
# Author: Jeonmin kang
# Mail: jarvisNim@gmail.com
# History
- 20231107  Create
- 20240209  OECD update 추가
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
import webbrowser
import shutil
import platform


'''
공통 영역
'''
# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')

# 검색기간 설정
_to_date = pd.to_datetime(to_date, format="%d/%m/%Y")
term_days = relativedelta(days=5)  # 초기 작업시는 12주 로 하면 사이트 부하오류 발생 안해서 최적! 평소에는 1주기간 데이터만으로도 가능
from_date = (_to_date - term_days).date()
to_date = _to_date.date()

download_directory = "./batch/reports/data"

'''
데이터 베이스와 거시 경제 테이블 생성
'''
# Connect DataBase
database = database_dir+'/'+'Economics.db'
conn, engine = create_connection(database)


# 테이블 정의

str_alpha = '(\
    Country TEXT NOT NULL,\
    Market TEXT NOT NULL,\
    Busi TEXT NOT NULL,\
    Researcher TEXT NOT NULL,\
    Date TEXT NOT NULL,\
    Country_Growth NUMERIC,\
    Market_Growth NUMERIC,\
    Busi_Growth NUMERIC,\
    Trend NUMERIC,\
    Trend_3mo NUMERIC,\
    Trend_6mo NUMERIC,\
    Trend_12mo NUMERIC,\
    Trend_18mo NUMERIC,\
    Trend_24mo NUMERIC,\
    PRIMARY KEY (Country, Market, Busi, Researcher, Date))'


str_calendars = '(\
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
        PRIMARY KEY(date,country,event))'

str_markets = '(\
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
    PRIMARY KEY (Country, Market, Symbol, Last_value))'

str_indicators = '(\
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
    Percentile INTEGER,\
    PRIMARY KEY (Country, Indicator, Date))'

str_stock_indices = '(\
    symbol TEXT NOT NULL,\
    name TEXT NOT NULL,\
    price NUMERIC,\
    changesPercentage NUMERIC,\
    change NUMERIC,\
    Previous NUMERIC,\
    dayLow NUMERIC,\
    dayHigh NUMERIC,\
    yearHigh NUMERIC,\
    yearLow NUMERIC,\
    marketCap NUMERIC,\
    priceAvg50 NUMERIC,\
    priceAvg200 NUMERIC,\
    exchange TEXT,\
    volume INTEGER,\
    avgVolume INTEGER,\
    open NUMERIC,\
    previousClose NUMERIC,\
    eps  NUMERIC,\
    pe   NUMERIC,\
    earningsAnnouncement    NUMERIC,\
    sharesOutstanding   NUMERIC,\
    timestamp INTEGER NOT NULL,\
    PRIMARY KEY (symbol, name, timestamp))'


'''
financemodeling.com 추출한 경제 캘린더 <=== 검증완료: 수기 작성과 어플리케이션 작성 모두 멀티 Primary key 구성 가능
'''
def create_Calendars(conn, str_calendar):
    with conn:
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE if not exists Calendars {str_calendars}')
    return conn

'''
MacroVar.com 에서 추출한 Financial Markets <== 현재는 사이트의 정보 업데이트가 미흡하여 미활용, 주기적으로 확인필요
'''
def create_Markets(conn, str_markets):
    with conn:
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE if not exists Markets {str_markets}')
    return conn

'''
MacroVar.com 에서 추출한 Macroeconomic Indicators
'''
def create_Indicators(conn, str_indicators):
    with conn:
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE if not exists Indicators {str_indicators}')

    return conn

'''
financialmodelingprep.com 에서 추출한 Stock Market Indices
'''
def create_Stock_Indices(conn, str_indicators):
    with conn:
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE if not exists Stock_Indices {str_stock_indices}')

    return conn

'''
각종 Research 에서 추출한 Trend, Country/Market/Business Growth 를 가지고 현 사이클 단계 확인과 6,12개월 전망
- Researcher: OECD, IMF, Tech, Senti >> Total
- Stauts: Buttom -> Bull-1 -> Bull-2 -> Top -> Bear-1 -> Bear-2
'''
def create_Alpha(conn, str_alpha):
    with conn:
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE if not exists Alpha {str_alpha}')
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
        logger.error(' >>> Exception: {}'.format(e))

# 테이블 데이터 insert
def write_table(table_name, data):
    insert_count = 0
    delete_count = 0

    if table_name == 'Calendars':
        data.dropna(subset=['date', 'country', 'event'], inplace=True)     
    elif table_name == 'Markets':
        data.dropna(subset=['Country', 'Market', 'Symbol', 'Last_value'], inplace=True) 
    elif table_name == 'Indicators':
        data.dropna(subset=['Country', 'Indicator', 'Date'], inplace=True)
    elif table_name == 'Stock_Indices':
        data.dropna(subset=['symbol', 'name', 'timestamp'], inplace=True)
    elif table_name == 'Alpha':
        data.dropna(subset=['Country', 'Market', 'Busi', 'Researcher', 'Date'], inplace=True)
    else:
        logger.error(' >>> Exception: Table Name Not found.')

    data = data.reset_index(drop=True) 

    # print(data[70:120])
    with conn:
        cur=conn.cursor()
        for i, row in data.iterrows():

            buf = data.iloc[i:i+1, :]
            if table_name == 'Calendars':
                _key = f"date = '{buf['date'][i]}' and country = '{buf['country'][i]}' and event = '{buf['event'][i]}'"     
            elif table_name == 'Markets':
                _key = f"Country = '{buf['Country'][i]}' and Symbol = '{buf['Symbol'][i]}' and Last_value = '{buf['Last_value'][i]}'"
            elif table_name == 'Indicators':
                _key = f"Country = '{buf['Country'][i]}' and Indicator = '{buf['Indicator'][i]}' and Date = '{buf['Date'][i]}'"
            elif table_name == 'Stock_Indices':
                _key = f"symbol = '{buf['symbol'][i]}' and name = '{buf['name'][i]}' and timestamp = '{buf['timestamp'][i]}'"
            elif table_name == 'Alpha':
                _key = f"Country = '{buf['Country'][i]}' and Market = '{buf['Market'][i]}' and Busi = '{buf['Busi'][i]}' and Researcher = '{buf['Researcher'][i]}' and Date = '{buf['Date'][i]}'"

            else:
                logger.error(' >>> Exception: Table Name Not found 2.')            

            # print(f'delete from {table_name} where {_key}')  # DEBUG
            cur.execute(f"delete from {table_name} where {_key}")
            cur.execute('commit')
            delete_count += 1

            # print('insert...', buf)     # DEBUG
            _cnt = buf.to_sql(table_name, con=engine, if_exists='append', chunksize=1000, index=False)

            insert_count += _cnt

    logger2.info('')
    logger2.info(f'{table_name} delete Count: ' + str({delete_count}))
    logger2.info(f'{table_name} insert Count: ' + str({insert_count}))
    logger2.info('')


# 덤프 테이블 데이터 replace
def write_dump_table(table_name, data):
    insert_count = 0

    _cnt = data.to_sql(table_name, con=engine, if_exists='replace', chunksize=1000, index=False, method='multi')
    
    insert_count += _cnt
    logger2.info('')    
    logger2.info(f'Dump {table_name} replace Count: ' + str({insert_count}))
    logger2.info('')    


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
    buf['Percentile'] = indicators['Percentile']

    return buf



'''
1. Calendars 테이블 데이터 구성
'''
def make_calendars(from_date, to_date):
    table_name = 'Calendars'
    cals = pd.DataFrame()
    for i in range(1):  # 최초 구성시는 20? 이후 매일 3회 배치작업으로 구성하고 있으니 1바퀴만 돌면 괜찮을듯.
        buf = get_calendar(from_date=from_date, to_date=to_date)
        buf = buf[buf['event'] != "New Year's Day"] # New Year's Day 제거, 's 로 syntax error 유발
        buf = buf[buf['event'] != "New Year's Eve"] # New Year's Eve 제거, 's 로 syntax error 유발 
        # print(buf)
        for i in range(len(NATIONS)):
            buf2 = buf[buf['country'] == NATIONS[i]]
            cals = pd.concat([cals, buf2], axis=0)
        to_date = pd.to_datetime(from_date)
        from_date = (to_date - term_days).date()
        to_date = to_date.date()

    cals = cals.reset_index(drop=True)

    logger2.info('')
    logger2.info(f' 최근 5일동안의 Calendars 중 estimate 까지 발표된 지표 '.center(60, '*'))
    buffer = cals[pd.to_datetime(cals['date']) >=  pd.to_datetime(to_date)]
    buffer = buffer.dropna(subset=['estimate'])  # 컨센서스가 지난 대비 긍정 또는 부정으로 판단하는지 볼수있는 estimate 도 포함.
    logger2.info(buffer.sort_values(by='date', ascending=False).to_string())    
    write_table(table_name, cals)


'''
2. Markets 테이블 데이터 구성
- macrovar.com 에서 markets 표 읽어오기
'''
def make_markets(**kwargs):
    table_name = 'Markets'
    df = pd.DataFrame()
    for key, value in kwargs.items():
        buf = get_markets(key, value, table_name)
        logger2.info('')
        logger2.info(f' macrovar.com 의 markets 표: {buf.Country[0]} '.center(60, '*'))
        logger2.info(buf)        
        df = pd.concat([df, buf])

    df = df.reset_index(drop=True)     
    write_table(table_name, df)

'''
3. Indicators 테이블 데이터 구성
- macrovar.com 에서 indicator 표 읽어오기
'''
def make_indicators(**kwargs):
    table_name = 'Indicators'
    df = pd.DataFrame()
    for key, value in kwargs.items():
        buf = get_indicators(key, value, table_name)
        logger2.info('')
        logger2.info(f'macrovar.com 의 indicator 표: {buf.Country[0]} '.center(60, '*'))
        logger2.info(buf)          
        df = pd.concat([df, buf])

    df = df.reset_index(drop=True)
    write_table(table_name, df)


'''
4. Global Stock Market Indices 테이블 데이터 구성
'''
def make_stock_indices(**kwargs):
    table_name = 'Stock_Indices'
    df = get_stock_indices()
    df = df.reset_index(drop=True)
    write_table(table_name, df)


'''
5. IMF Data 구성
https://www.imf.org/en/Publications/SPROLLS/world-economic-outlook-databases#sort=%40imfdate%20descending 사이트에서 
최신 업데이트 정보목록을 선택하여 들어간후, Entire Dataset 버튼 > By Country 버튼 누르면 다운로드 완료.
Download 사이트에서 해당 파일의 이름을 IMF_outlook.xls 로 변경후 batch/reports/data/으로 이동한후 DB 업로드 프로그램 수행
- 20240103 https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2023/WEOOct2023all.ashx 
- 20240131 현재 https://www.imf.org/-/media/Files/Publications/WEO/2024/Update/January/English/Data/WEOJAN2024Update.ashx
'''
def make_imf_outlook():

    table_name = 'IMF'

    # url = 'https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2023/WEOOct2023all.ashx'
    url = 'https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2023/WEOOct2023all.ashx'    

    response = requests.get(url, timeout=10, verify=False)
    
    if response.status_code == 200:
        with open(download_directory+'/IMF.csv', 'wb') as file:
            file.write(response.content)
        sleep(10)            
    else:
        logger.error(f" >>> WorldBank.xlsx 다운로드 실패. 응답 코드: {response.status_code}")

    df = pd.read_csv('./batch/reports/data/IMF.csv', sep='\t', skip_blank_lines=True, skipfooter=3, encoding_errors='replace')
    df = df.reset_index(drop=True)
    write_dump_table(table_name, df)



'''
6. OECD Data 구성
- ref: https://medium.com/@akshaybagal/oecd-stats-website-your-go-to-for-comprehensive-statistics-datasets-on-oecd-countries-2ae04e4aa044
- OECD 아래 사이트 들어가서 > Export > Developer API > 'Generate API Queries' 버튼 눌러서 나온 url 으로 구성, # 2025년 까지를 2030년으로 길게 늘려둠. ^^
  https://stats.oecd.org/Index.aspx?DataSetCode=EO
- id_to_name_mappings:
{'Country': {'AUS': 'Australia',
  'AUT': 'Austria',
  'BEL': 'Belgium',
  'CAN': 'Canada',
  'CZE': 'Czechia',
  'DNK': 'Denmark',
  'FIN': 'Finland',
  'FRA': 'France',
  'DEU': 'Germany',
  'GRC': 'Greece',
  'HUN': 'Hungary',
  'ISL': 'Iceland',
  'IRL': 'Ireland',
  'ITA': 'Italy',
  'JPN': 'Japan',
  'KOR': 'Korea',
  'LUX': 'Luxembourg',
  'MEX': 'Mexico',
  'NLD': 'Netherlands',
  'NZL': 'New Zealand',
  'NOR': 'Norway',
  'POL': 'Poland',
  'PRT': 'Portugal',
  'SVK': 'Slovak Republic',
  'ESP': 'Spain',
  'SWE': 'Sweden',
  'CHE': 'Switzerland',
  'TUR': 'Türkiye',
  'GBR': 'United Kingdom',
  'USA': 'United States',
  'OTO': 'OECD - Total',
  'BRA': 'Brazil',
  'CHL': 'Chile',
  'CHN': "China (People's Republic of)",
  'EST': 'Estonia',
  'IND': 'India',
  'IDN': 'Indonesia',
  'ISR': 'Israel',
  'RUS': 'Russia',
  'SVN': 'Slovenia',
  'ZAF': 'South Africa',
  'WLD': 'World',
  'DAE': 'Dynamic Asian Economies',
  'OOP': 'Other oil producers',
  'COL': 'Colombia',
  'LVA': 'Latvia',
  'NMEC': 'Non-OECD Economies',
  'LTU': 'Lithuania',
  'CRI': 'Costa Rica',
  'ARG': 'Argentina',
  'ROU': 'Romania',
  'BGR': 'Bulgaria',
  'HRV': 'Croatia',
  'PER': 'Peru'},
 'Variable': {'CBGDPR': 'Current account balance as a percentage of GDP',
  'EXCHUD': 'Exchange rate, national currency per USD',
  'MGSVD': 'Imports of goods and services, volume in USD (national accounts basis)',
  'TGSVD': 'Goods and services trade, volume in USD',
  'XGSVD': 'Exports of goods and services, volume in USD (national accounts basis)',
  'NLGQ': 'General government net lending as a percentage of GDP',
  'GGFLQ': 'General government gross financial liabilities as a percentage of GDP',
  'GDP': 'Gross domestic product, nominal value, market prices',
  'GDPV': 'Gross domestic product, volume, market prices',
  'ITV': 'Gross fixed capital formation, total, volume',
  'CGV': 'Government final consumption expenditure, volume',
  'CPV': 'Private final consumption expenditure, volume',
  'XGSV_ANNPCT': 'Exports of goods and services, volume, growth (national accounts basis)',
  'MGSV_ANNPCT': 'Imports of goods and services, volume, growth (national accounts basis)',
  'PGDP_ANNPCT': 'Gross domestic product, market prices, deflator, growth',
  'PCORE_YTYPCT': 'Core inflation',
  'CPV_ANNPCT': 'Private final consumption expenditure, volume, growth',
  'CPI_YTYPCT': 'Headline inflation',
  'CGV_ANNPCT': 'Government final consumption expenditure, volume, growth',
  'ITV_ANNPCT': 'Gross fixed capital formation, total, volume, growth',
  'GDPV_ANNPCT': 'Gross domestic product, volume, growth',
  'CPI': 'Consumer price index',
  'PXGS': 'Exports of goods and services, deflator (national accounts basis)',
  'PMGS': 'Imports of goods and services, deflator (national accounts basis)',
  'PGDP': 'Gross domestic product, market prices, deflator',
  'UNR': 'Unemployment rate',
  'ET': 'Total employment (labour force survey basis)',
  'LF': 'Labour force',
  'IRL': 'Long-term interest rate on government bonds',
  'IRS': 'Short-term interest rate',
  'GGFLMQ': 'Gross public debt, Maastricht criterion as a percentage of GDP',
  'PCOREH_YTYPCT': 'Harmonised core inflation',
  'CPIH_YTYPCT': 'Harmonised headline inflation',
  'CPIH': 'Consumer price index, harmonised',
  'WPBRENT': 'Crude oil price, FOB, USD per barrel, spot Brent',
  'PCOREH': 'Core inflation index, harmonised',
  'MGSV': 'Imports of goods and services, volume (national accounts basis)',
  'PCORE': 'Core inflation index',
  'XGSV': 'Exports of goods and services, volume (national accounts basis)',
  'CQ_FBGSV': 'Net exports, contributions to changes in real GDP',
  'GDPVD': 'Gross domestic product, volume in USD, at constant purchasing power parities',
  'GDP_ANNPCT': 'Gross domestic product, nominal value, growth',
  'EXCH': 'Exchange rate, USD per national currency',
  'YDH': 'Net disposable income of households and non-profit institutions serving households',
  'NLGXQ': 'General government primary balance as a percentage of GDP',
  'PPP': 'Purchasing power parity, national currency per USD',
  'CQ_ISKV': 'Change in inventories, contributions to changes in real GDP',
  'ITISKV': 'Gross capital formation, total, volume',
  'GFAR': 'General government gross financial assets as a percentage of GDP',
  'YDH_G': 'Gross disposable income of household and non-profit institutions serving households',
  'GDPML': 'Mainland gross domestic product , nominal value, market prices',
  'GDPMLV': 'Mainland gross domestic product, volume, market prices',
  'GDPV_USD': 'Gross domestic product, volume in USD, constant exchange rates'},
 'Frequency': {'A': 'Annual'},
 'Time': {'2016': '2016',
  '2017': '2017',
  '2018': '2018',
  '2019': '2019',
  '2020': '2020',
  '2021': '2021',
  '2022': '2022',
  '2023': '2023',
  '2024': '2024',
  '2025': '2025'}}

6.1 Interim report update function: update_oecd_interim()
'''
def make_oecd_outlook():

    table_name = 'OECD'

    url = "https://stats.oecd.org/SDMX-JSON/data/EO/AUS+AUT+BEL+CAN+CHL+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA+OTO+WLD+NMEC+ARG+BRA+BGR+CHN+HRV+IND+IDN+PER+ROU+RUS+ZAF+DAE+OOP.EXT+CBGDPR+EXCHUD+EXCH+XGSVD+TGSVD+MGSVD+GOV+GFAR+GGFLQ+NLGQ+NLGXQ+GGFLMQ+EXP+CQ_ISKV+XGSV+CGV+ITISKV+GDP+GDPVD+GDPV_USD+GDPV+ITV+MGSV+GDPML+GDPMLV+CQ_FBGSV+CPV+SEL+PCORE_YTYPCT+XGSV_ANNPCT+CGV_ANNPCT+PGDP_ANNPCT+GDP_ANNPCT+GDPV_ANNPCT+ITV_ANNPCT+PCOREH_YTYPCT+CPIH_YTYPCT+CPI_YTYPCT+MGSV_ANNPCT+CPV_ANNPCT+HOU+YDH_G+YDH+PRI+CPI+CPIH+PCORE+PCOREH+PXGS+PGDP+PMGS+PPP+LAB+LF+ET+UNR+MON+IRL+IRS+OIL+WPBRENT.A/all?startTime=2016&endTime=2030&dimensionAtObservation=allDimensions"
    response = requests.request("GET", url)

    data = response.json()
    data_values = data['dataSets'][0]['observations']
    dimensions = data['structure']['dimensions']['observation']

    id_to_name_mappings = {
        dim['name']: {item['id']: item['name'] for item in dim['values']}
        for dim in dimensions
    }

    dimension_values = [dim['values'] for dim in dimensions]

    def get_id_from_index(dim_index, index):
        return dimension_values[dim_index][index]['id']

    def map_id_to_name(dim_name, id):
        return id_to_name_mappings[dim_name].get(id, id)

    rows = []
    for key, value in data_values.items():
        indices = key.split(':')  # Split keys into separate dimension indices
        country = map_id_to_name('Country', get_id_from_index(0, int(indices[0])))
        variable = map_id_to_name('Variable', get_id_from_index(1, int(indices[1])))
        frequency = map_id_to_name('Frequency', get_id_from_index(2, int(indices[2])))  # Corrected typo here
        time = map_id_to_name('Time', get_id_from_index(3, int(indices[3])))
        data_value = value[0]  # Extract the data value
        rows.append([country, variable, frequency, time, data_value])  # Append the row to the list of rows

    df = pd.DataFrame(rows, columns=['Country', 'Variable', 'Frequency', 'Time', 'Value'])
    write_dump_table(table_name, df)

    return df


'''
대상 indicator 는 inflation, gdp 
- 20240207: OECD Economic Outlook, Interim Report February 2024: inflation, gdp
- SELECT * FROM OECD WHERE Variable like '%Gross domestic product, volume, growth%' AND (Time = '2024' or Time = '2025')
- SELECT * FROM OECD WHERE Variable like '%Headline inflation%' AND (Time = '2024' or Time = '2025')
'''
def update_oecd_interim(conn):
    variable_gdp = "%Gross domestic product, volume, growth%"  # Source: OECD Economic Outlook, Interim Report February 2024.
    variable_inflation = "%Headline inflation%"  # Headline Inflation, Source: OECD Economic Outlook, Interim Report February 2024.
    gdp_update =     {
        'Brazil%':[{'2024':1.81771723436261}, {'2025':2.02464923513675}],
        'China%': [{'2024':4.68607906868843}, {'2025':4.24280944230372}],
        'Euro%': [{'2024':0.635063372653745}, {'2025':1.33172323471308}],        
        'Germany%': [{'2024':0.279680495326152}, {'2025':1.10892974166749}],
        'India%': [{'2024':6.15974810807787}, {'2025':6.51778646588327}],
        'Japan%': [{'2024':0.997774345949994}, {'2025':0.998694183827788}],
        'Korea%': [{'2024':2.16269397354112}, {'2025':2.11686859085889}],
        'United States%': [{'2024':2.1488822258862}, {'2025':1.71411734194476}],
    }

    inflation_update =     {
        'Brazil%':[{'2024':3.28414019019374}, {'2025':2.99603206276178}],
        'China%': [{'2024':1.04729033978783}, {'2025':1.45639190001296}],
        'Euro%': [{'2024':0.635063372653745}, {'2025':1.33172323471308}],        
        'Germany%': [{'2024':2.58814674906734}, {'2025':2.01387974466703}],
        'India%': [{'2024':4.91947232367986}, {'2025':4.25901301129225}],
        'Japan%': [{'2024':2.59695374744398}, {'2025':2.04680754163396}],
        'Korea%': [{'2024':2.66138758411523}, {'2025':2.03949651385258}],
        'United States%': [{'2024':2.17034237341851}, {'2025':1.9958871662692}],
    }    

    with conn:
        cur=conn.cursor()
        update_count = 0
        for country in gdp_update.keys():
            # print(x)
            # print(oecd_update[x])
            for i, gdp in enumerate(gdp_update[country]):
                key = list(gdp.keys())
                year = key[0]
                value = gdp[key[0]]
                # print(year)
                # print(value)
                if i == 0:
                    second_year = str(int(year)+1)
                    # print(f"SELECT * FROM OECD WHERE Country like '{country}' AND Variable like '{variable_gdp}' AND (Time = '{year}' or Time = '{second_year}') ")
                    cur.execute(f"SELECT * FROM OECD WHERE Country like '{country}' AND Variable like '{variable_gdp}' AND (Time = '{year}' or Time = '{second_year}') ")

                # print(f"update OECD set Value = {value} where Country like '{country}' AND Variable like '{variable_gdp}' and Time = {year} ")
                cur.execute(f"update OECD set Value = {value} where Country like '{country}' AND Variable like '{variable_gdp}' and Time = {year} ")
                update_count += 1
        logger2.info(f"GDP Growth update count: {update_count}") 

        # Inflation update
        update_count = 0
        for country in inflation_update.keys():
            # print(x)
            # print(oecd_update[x])
            for i, inflation in enumerate(inflation_update[country]):
                key = list(inflation.keys())
                year = key[0]
                value = inflation[key[0]]
                # print(year)
                # print(value)
                if i == 0:
                    second_year = str(int(year)+1)
                    # print(f"SELECT * FROM OECD WHERE Country like '{country}' AND Variable like '{variable_inflation}' AND (Time = '{year}' or Time = '{second_year}') ")
                    cur.execute(f"SELECT * FROM OECD WHERE Country like '{country}' AND Variable like '{variable_inflation}' AND (Time = '{year}' or Time = '{second_year}') ")

                # print(f"update OECD set Value = {value} where Country like '{country}' AND Variable like '{variable_inflation}' and Time = {year} ")
                cur.execute(f"update OECD set Value = {value} where Country like '{country}' AND Variable like '{variable_inflation}' and Time = {year} ")
                update_count += 1
        logger2.info(f"Inflation update count: {update_count}")            
        
        cur.execute('commit')
        


'''
7. WorldBank Data 생성
https://www.worldbank.org/en/news/press-release/2024/01/09/global-economic-prospects-january-2024-press-release?intcid=ecr_hp_headerA_2024-01-09-GEPPressRelease
사이트에서  (향후 하반기 업데이트시는 다른 사이트일듯.), 'Download growth data' 버튼 누르면 다운로드 완료.
Download 사이트에서 해당 파일의 이름을 WorldBank.xls 로 변경후 batch/reports/data/으로 이동한후 DB 업로드 프로그램 수행
- 20240110 현재 https://bit.ly/GEP-Jan-2024-GDP-growth-data 파일이 최신임
'''
def make_worldbank_outlook():

    table_name = 'WorldBank'

    url = 'https://bit.ly/GEP-Jan-2024-GDP-growth-data'

    response = requests.get(url, timeout=10, verify=False)
    
    if response.status_code == 200:
        with open(download_directory+'/WorldBank.xlsx', 'wb') as file:
            file.write(response.content)
        sleep(10)            
    else:
        logger.error(f" >>> WorldBank.xlsx 다운로드 실패. 응답 코드: {response.status_code}")

    df = pd.read_excel('./batch/reports/data/WorldBank.xlsx', skiprows=range(0, 3),)
    df = df.reset_index(drop=True)
    df.columns = ['Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5', '_2021', '_2022', '_2023e',\
                  '_2024f', '_2025f', 'filler' ,'_2023e_d', '_2024f_d', '_2025f_d']
    write_dump_table(table_name, df)



'''
8. Alpha 테이블 생성
glbal_.py 프로그램 수행하면서 발생되는 국가/시장/사업별로 저장하는 루틴
각종 Researcher 에서 추출한 Trend, Country/Market/Business Growth 를 가지고 현 사이클 단계 확인과 6,12, 18, 24개월 전망 지표 보여줌.
- Researcher: OECD, IMF, WorldBank, Tech.., Senti.. >> Total
'''
def make_alpha(data):
    table_name = 'Alpha'
    # print(data)
    df = data
    df = df.dropna()
    df = df.reset_index(drop=True)
    write_table(table_name, df)


'''
999. 주기적 insert/delete 작업으로 키값 저장위치가 분산되어 그래프화시 노이즈 발생하여 이를 제거하기 위하여 재구성함.
- cron 작업 첫번째로 순서 변경
'''
def reorg_tables(conn):

    cur = conn.cursor()

    # 테이블 목록 조회
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    for table in tables:
        logger2.info(f'Reorg table names: {table[0]}')
    logger2.info('')


    # 1. Markets 테이블 재구성
    try:
        cur.execute('DROP TABLE Markets_backup;')
    except Exception as e:
        logger.error(' >>> Exception: {}'.format(e))
        pass
    cur.execute(f'CREATE TABLE Markets_backup {str_markets};')
    cur.execute('INSERT INTO Markets_backup SELECT * FROM Markets ORDER BY Date;')    
    cur.execute('DROP TABLE Markets;')

    cur.execute(f'CREATE TABLE Markets {str_markets};')
    cur.execute(f'INSERT INTO Markets SELECT * FROM Markets_backup;')    
    cur.execute('SELECT count(*) FROM Markets;')
    result = cur.fetchone()      
    logger2.info(f'Markets Reorg Count: ' + str(result[0]))    # result에는 (행의 수,) 형태의 튜플이 들어 있습니다.
    conn.commit()

    # 2. Indicators 테이블 재구성    
    try:
        cur.execute('DROP TABLE Indicators_backup;')
    except Exception as e:
        logger.error(' >>> Exception: {}'.format(e))
        pass
    cur.execute(f'CREATE TABLE Indicators_backup {str_indicators};')
    cur.execute('INSERT INTO Indicators_backup SELECT * FROM Indicators ORDER BY Country, Indicator, Date;')    
    cur.execute('DROP TABLE Indicators;')

    cur.execute(f'CREATE TABLE Indicators {str_indicators};')
    cur.execute(f'INSERT INTO Indicators SELECT * FROM Indicators_backup;')    
    cur.execute('SELECT count(*) FROM Indicators;')
    result = cur.fetchone()      
    logger2.info(f'Indicators Reorg Count: ' + str(result[0]))    # result에는 (행의 수,) 형태의 튜플이 들어 있습니다.
    conn.commit()

    # 3. Calendars 테이블 재구성
    try:
        cur.execute('DROP TABLE Calendars_backup;')
    except Exception as e:
        logger.error(' >>> Exception: {}'.format(e))
        pass
    cur.execute(f'CREATE TABLE Calendars_backup {str_calendars};')
    cur.execute('INSERT INTO Calendars_backup SELECT * FROM Calendars ORDER BY country, date;')    
    cur.execute('DROP TABLE Calendars;')

    cur.execute(f'CREATE TABLE Calendars {str_calendars};')
    cur.execute(f'INSERT INTO Calendars SELECT * FROM Calendars_backup;')    
    cur.execute('SELECT count(*) FROM Calendars;')
    result = cur.fetchone()      
    logger2.info(f'Calendars Reorg Count: ' + str(result[0]))    # result에는 (행의 수,) 형태의 튜플이 들어 있습니다.
    conn.commit()

    # 4. Stock_Indices 테이블 재구성
    try:
        cur.execute('DROP TABLE Stock_Indices_backup;')
    except Exception as e:
        logger.error(' >>> Exception: {}'.format(e))
        pass
    cur.execute(f'CREATE TABLE Stock_Indices_backup {str_stock_indices};')
    cur.execute('INSERT INTO Stock_Indices_backup SELECT * FROM Stock_Indices ORDER BY symbol, name, timestamp;')    
    cur.execute('DROP TABLE Stock_Indices;')

    cur.execute(f'CREATE TABLE Stock_Indices {str_stock_indices};')
    cur.execute(f'INSERT INTO Stock_Indices SELECT * FROM Stock_Indices_backup;')    
    cur.execute('SELECT count(*) FROM Stock_Indices;')
    result = cur.fetchone()      
    logger2.info(f'Stock_Indices Reorg Count: ' + str(result[0]))    # result에는 (행의 수,) 형태의 튜플이 들어 있습니다.
    conn.commit()    

    return conn


'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    # 테이블 생성 (최초 생성시만 Active 해서 사용 !!!)
    create_Alpha(conn, str_alpha)
    create_Calendars(conn, str_calendars)
    create_Markets(conn, str_markets)
    create_Indicators(conn, str_indicators)
    create_Stock_Indices(conn, str_stock_indices)
    '''

    '''
    # 테이블내 데이터 만들어 넣기
    '''
    try:
        make_calendars(from_date, to_date)
        make_markets(**urls)
        make_indicators(**urls)
        make_stock_indices()
        make_imf_outlook()
        make_oecd_outlook()
        update_oecd_interim(conn)
        make_worldbank_outlook()
    except Exception as e:
        logger.error(' >>> Exception: {}'.format(e))

    '''
    # make_alpha() 는 존재하지 않음.  _global.py 프로그램에서 호출해서 수행함.   
    '''



    '''
    # 테이블 저장공간 키구성순을 위한 재구성작업
    '''
    try:
        reorg_tables(conn)
    except Exception as e:
        logger.error(' >>> Exception: {}'.format(e))









