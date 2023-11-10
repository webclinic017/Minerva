'''
Prgram 명: Glance of Germany
Author: jimmy Kang
Mail: jarvisNim@gmail.com
독일 투자를 위한 경제지표(거시/미시) 부문 엿보기
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
logger.warning(sys.argv[0])
logger2.info(sys.argv[0])

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
to_date_2 = pd.to_datetime(today)
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
    M_country = 'DE'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        cals = pd.read_sql_query(M_query, conn)
        logger2.info(cals[:30])
    except Exception as e:
        print('Exception: {}'.format(e))

    events = ['Harmonised Inflation Rate YoY ', '30-Year Bund/g Auction', 'Import Prices MoM ', 'Brandenburg CPI MoM ', \
            'Ifo Current Conditions ', '5-Year Bobl Auction', 'Inflation Rate MoM ', 'Current Account ', \
            '10-Year Bund/g Auction', 'Industrial Production MoM ', 'Exports MoM ', '15-Year Bund Auction', \
            'ZEW Current Conditions ', 'GDP Growth Rate QoQ ', 'Brandenburg CPI YoY ', 'Inflation Rate YoY ', \
            'New Car Registrations YoY ', '30-Year Bund Auction', 'PPI MoM ', '30-Year Bund/€i Auction', \
            'HCOB Composite PMI ', 'Factory Orders MoM ', 'Bavaria CPI YoY ', 'HCOB Construction PMI ', \
            '10-Year Bund Auction', 'Retail Sales MoM ', 'Retail Sales YoY ', 'Hesse CPI MoM ', 'Balance of Trade ', \
            'PPI YoY ', 'Saxony CPI YoY ', 'Bund/g Auction', 'HCOB Manufacturing PMI ', 'ZEW Economic Sentiment Index ', \
            'Bavaria CPI MoM ', 'Saxony CPI MoM ', 'North Rhine Westphalia CPI MoM ', 'GDP Growth Rate YoY ', \
            'North Rhine Westphalia CPI YoY ', 'CPI ', 'Unemployment Rate ', '3-Month Bubill Auction', 'Hesse CPI YoY ', \
            'Harmonised Inflation Rate MoM ', '7-Year Bund Auction', 'Gfk Consumer Confidence ', 'GfK Consumer Confidence ', \
            'Wholesale Prices YoY ', '6-Month Bubill Auction', 'Baden Wuerttemberg CPI MoM ', 'Imports MoM ', \
            '2-Year Schatz Auction', 'Unemployed Persons ', 'Ifo Expectations ', 'Index-Linked Bund Auction', \
            'Import Prices YoY ', '10-Year Bund/€i Auction', 'Wholesale Prices MoM ', 'Unemployment Change ', \
            '12-Month Bubill Auction', '9-Month Bubill Auction', 'Baden Wuerttemberg CPI YoY ', 'Ifo Business Climate ', \
            'HCOB Services PMI ', ]
    
     # 전체 그림의 크기를 설정
    plt.figure(figsize=(18, 4*len(events)))
    for i, event in enumerate(events):
        result = cals[cals['event'].str.contains(event, case=False, na=False)]
        result['date'] = result['date'].apply(parse_date)
        plt.subplot(len(events), 1, i + 1)
        plt.plot(result['date'], result['actual'])
        plt.title(event)
        plt.xlabel('date')
        plt.ylabel('actual')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/germany_0000.png')

    return cals










'''
Main Fuction
'''

if __name__ == "__main__":
    cals = eco_calendars(from_date, to_date_2)  # calendars
    # shanghai_shares, szse_shares = shanghai_szse_vs_yuan(from_date_MT, to_date)
    # shanghai_vs_loan(cals)
    # shanghai_vs_m2(cals)
    # house_loan(cals)
    # yuan_exchange_rate(from_date_MT)
    # gdp_yoy, gdp_qoq = shanghai_vs_gpd(cals)
    # shanghai_vs_ip(cals, gdp_yoy, gdp_qoq)
    # shanghai_vs_house(cals)
    # shanghai_vs_eximport(cals)
    # shanghai_vs_dollar(cals)
    # shanghai_vs_cpi_ppi(cals)   
    # pmi(cals)
    # indu_profit(cals)
    # foreign_invest(cals)
    # fixed_asset_invest(cals)