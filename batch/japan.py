'''
Prgram 명: Glance of Japan
Author: jimmy Kang
Mail: jarvisNim@gmail.com
일본 투자를 위한 경제지표(거시/미시) 부문 엿보기
* investing.com/calendar 포함
History
20231031  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

import requests
from bs4 import BeautifulSoup as bs
import yfinance as yf

'''
시작/종료 일자 셋팅
'''
to_date_2 = pd.to_datetime(today)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()

# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0])


'''
경제지표 그래프
'''
def eco_indexes(from_date, to_date):
    to_date_2 = pd.to_datetime(today)
    three_month_days = relativedelta(weeks=12)
    from_date = (to_date_2 - three_month_days).date()
    to_date_2 = to_date_2.date()
    cals = pd.DataFrame()

    for i in range(10):  # 10 -> 2 for test
        buf = get_calendar(from_date=from_date, to_date=to_date_2)
        buf = buf[buf['country'] == 'JP']
        cals = pd.concat([cals, buf], axis=0)
        to_date_2 = pd.to_datetime(from_date)
        from_date = (to_date_2 - three_month_days).date()
        to_date_2 = to_date_2.date()

    temp = pd.to_datetime(cals.index)
    temp2 = temp.date
    cals.set_index(temp2, inplace=True)
    cals.index.name = 'date'
    logger2.info(cals[:30])

    # buf = pd.DataFrame()
    # l = []
    # for x in cals['event']:
    #     t = x.split('(')[0]
    #     l.append(t)
    # unique_list = list(set(l))
    # print(unique_list)

    events = ['Eco Watchers Survey Current ', 'GDP Price Index YoY ', 'GDP External Demand QoQ ', 'Exports YoY ', 'Eco Watchers Survey Outlook ', 'Capital Spending YoY ', 'BSI Large Manufacturing QoQ ', 'Leading Economic Index ', 'Average Cash Earnings YoY ', 'GDP Growth Annualized ', 'Tokyo Core CPI YoY ', 'Coincident Index ', 'Tertiary Industry Index MoM ', 'Inflation Rate MoM ', 'Tankan Large Manufacturing Outlook ', 'Jobs/applications ratio ', 'Stock Investment by Foreigners ', 'Inflation Rate Ex-Food and Energy YoY ', 'Housing Starts YoY ', 'Household Spending YoY ', 'Core Inflation Rate YoY ', 'Tankan Large All Industry Capex ', 'PPI MoM ', 'Overtime Pay YoY ', 'Machinery Orders YoY ', 'Jibun Bank Services PMI ', 'Tankan Large Non-Manufacturing Index ', 'Current Account ', 'BoJ Interest Rate Decision', 'Unemployment Rate ', 'Industrial Production MoM ', 'Reuters Tankan Index ', 'Imports YoY ', '5 Year Note Yield', 'Foreign Exchange Reserves ', '5-Year JGB Auction', '52-Week Bill Auction', 'Tankan Non-Manufacturing Outlook ', 'GDP Growth Rate QoQ ', 'Machinery Orders MoM ', 'Foreign Bond Investment ', 'Household Spending MoM ', 'Jibun Bank Manufacturing PMI ', 'Industrial Production MoM. ', 'GDP Capital Expenditure QoQ ', 'Balance of Trade ', '6-Month Bill Auction', 'Jibun Bank Composite PMI ', '40-Year JGB Auction', '5-Year Note Yield', 'Industrial Production YoY ', 'PPI YoY ', 'Retail Sales YoY ', 'Inflation Rate YoY ', '30-Year JGB Auction', 'Tankan Large Manufacturers Index ', 'Tokyo CPI YoY ', 'Tankan Small Manufacturers Index ', '2 Year Note Yield', 'Bank Lending YoY ', 'Consumer Confidence ', '2-Year JGB Auction', '3-Month Bill Auction', 'Machine Tool Orders YoY ', 'BoJ Nakamura Spech', 'GDP Private Consumption QoQ ', 'Retail Sales MoM ', 'Tokyo CPI Ex Food and Energy YoY ', 'Construction Orders YoY ', 'Capacity Utilization MoM ']

    for i, event in enumerate(events):
        
        result = cals[cals['event'].str.contains(event, case=False, na=False)]
        plt.subplot(len(events), 1, i + 1)
        result['actual'].plot(figsize=(18,300), title=event)    
        plt.xlabel('Date')
        plt.ylabel('actual')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/japan_0000.png')

    return cals










'''
Main Fuction
'''

if __name__ == "__main__":
    cals = eco_indexes(from_date, to_date_2)  # calendars
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