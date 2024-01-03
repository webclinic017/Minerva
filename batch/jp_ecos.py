'''
Prgram 명: Glance of Japan
Author: jimmy Kang
Mail: jarvisNim@gmail.com
일본 투자를 위한 경제지표(거시/미시) 부문 엿보기
* investing.com/calendar 포함
History
20231031  Create
20231110  Economics 테이블 쿼리문으로 변경, 기존 url 읽기에서.
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

import requests
from bs4 import BeautifulSoup as bs
import yfinance as yf

'''
0. 공통영역 설정
'''

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
to_date_2 = pd.to_datetime(to_date2)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()

# Connect DataBase
database = database_dir+'/'+'Economics.db'
conn, engine = create_connection(database)


'''
경제지표 그래프
'''
def eco_calendars(from_date, to_date):

    # 날짜 및 시간 문자열을 날짜로 변환하는 함수
    def parse_date(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()

    # 2) Economics database 에서 쿼리후 시작하는 루틴
    M_table = 'Calendars'
    M_country = 'JP'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        cals = pd.read_sql_query(M_query, conn)
        logger2.info('=== Calendars ===')
        logger2.info(cals[-30:])
    except Exception as e:
        print('Exception: {}'.format(e))

    events = ['Eco Watchers Survey Current ', 'GDP Price Index YoY ', 'GDP External Demand QoQ ', 'Exports YoY ', \
            'Eco Watchers Survey Outlook ', 'Capital Spending YoY ', 'BSI Large Manufacturing QoQ ', 'Leading Economic Index ', \
            'Average Cash Earnings YoY ', 'GDP Growth Annualized ', 'Tokyo Core CPI YoY ', 'Coincident Index ', 'Tertiary Industry Index MoM ', \
            'Inflation Rate MoM ', 'Tankan Large Manufacturing Outlook ', 'Jobs/applications ratio ', 'Stock Investment by Foreigners ', \
            'Inflation Rate Ex-Food and Energy YoY ', 'Housing Starts YoY ', 'Household Spending YoY ', 'Core Inflation Rate YoY ', \
            'Tankan Large All Industry Capex ', 'PPI MoM ', 'Overtime Pay YoY ', 'Machinery Orders YoY ', 'Jibun Bank Services PMI ', \
            'Tankan Large Non-Manufacturing Index ', 'Current Account ', 'BoJ Interest Rate Decision', 'Unemployment Rate ', \
            'Industrial Production MoM ', 'Reuters Tankan Index ', 'Imports YoY ', '5 Year Note Yield', 'Foreign Exchange Reserves ', \
            '5-Year JGB Auction', '52-Week Bill Auction', 'Tankan Non-Manufacturing Outlook ', 'GDP Growth Rate QoQ ', 'Machinery Orders MoM ', \
            'Foreign Bond Investment ', 'Household Spending MoM ', 'Jibun Bank Manufacturing PMI ', 'Industrial Production MoM. ', \
            'GDP Capital Expenditure QoQ ', 'Balance of Trade ', '6-Month Bill Auction', 'Jibun Bank Composite PMI ', '40-Year JGB Auction', \
            '5-Year Note Yield', 'Industrial Production YoY ', 'PPI YoY ', 'Retail Sales YoY ', 'Inflation Rate YoY ', '30-Year JGB Auction', \
            'Tankan Large Manufacturers Index ', 'Tokyo CPI YoY ', 'Tankan Small Manufacturers Index ', '2 Year Note Yield', \
            'Bank Lending YoY ', 'Consumer Confidence ', '2-Year JGB Auction', '3-Month Bill Auction', 'Machine Tool Orders YoY ', \
            'BoJ Nakamura Spech', 'GDP Private Consumption QoQ ', 'Retail Sales MoM ', 'Tokyo CPI Ex Food and Energy YoY ', \
            'Construction Orders YoY ', 'Capacity Utilization MoM ']

    # 전체 그림의 크기를 설정
    plt.figure(figsize=(10, 3*len(events)))
    for i, event in enumerate(events):
        result = cals[cals['event'].str.contains(event, case=False, na=False)]
        result['date'] = result['date'].apply(parse_date)
        plt.subplot(len(events), 1, i + 1)
        plt.plot(result['date'], result['actual'])
        plt.title(event)
        plt.xlabel('date')
        plt.ylabel('actual')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/jp_e0000.png')

    return cals










'''
Main Fuction
'''

if __name__ == "__main__":
    cals = eco_calendars(from_date, to_date_2)  # calendars
