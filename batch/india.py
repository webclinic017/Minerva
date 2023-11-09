'''
Prgram 명: Glance of India
Author: jimmy Kang
Mail: jarvisNim@gmail.com
일본 투자를 위한 경제지표(거시/미시) 부문 엿보기
*인도는 과연 중국을 이을 다음 글로벌 공장이 될수 있을 것인가 ?
History
20221101  Create
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
        buf = buf[buf['country'] == 'IN']
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
    plt.savefig(reports_dir + '/india_0000.png')

    return cals


'''
1. 통화/금융
1.1 BSE(Bombay Stock Exchange) vs INR
'''
def bse_vs_inr(from_date, to_date):
    # BSE(Bombay Stock Exchange) + NSE(National Stock Exchange)
    india_shares = fred.get_series(series_id='SPASTT01INM661N', observation_start=from_date_MT)

    # Indian Rupees
    rupees = fred.get_series(series_id='EXINUS', observation_start=from_date_MT)

    # India GDP by Expenditure
    india_growth = fred.get_series(series_id='NAEXKP03INQ659S', observation_start=from_date_MT)

    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(india_shares, label='BSE + NSE', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(india_growth, label='Growth Rate', linewidth=1, linestyle='-', color='orange')
    ax2.axhline(y=0, linestyle='--', color='red', linewidth=2)
    plt.title(f"India Shares vs Growth Rate", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/india_0110.png')

    # BSE(Bombay Stock Exchange) + NSE(National Stock Exchange)
    india_shares = fred.get_series(series_id='SPASTT01INM661N', observation_start=from_date_MT)

    # Indian Rupees
    rupees = fred.get_series(series_id='EXINUS', observation_start=from_date_MT)

    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(india_shares, label='BSE + NSE', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(rupees, label='USD/INR', linewidth=1, linestyle='-', color='orange')
    plt.title(f"India Shares vs Indian Rupees", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/india_0111.png')

    return india_shares, rupees


'''
1.2 Shares vs PBoC Loan Prime Rate
'''
def shares_vs_loan(cals):
    # Bank Loan Growth
    buf = cals.loc[cals['event'].str.contains('Bank Loan Growth')]
    logger2.info(buf[-7::2])

    # Graph
    fig, ax1 = plt.subplots(figsize=(15,5))
    lns1 = ax1.plot(buf.index, buf['actual'], label='Bank Loan Growth', linewidth=2,\
                    linestyle='-', marker='x', color='maroon')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(india_shares, label='india_shares', linewidth=1, linestyle='-')
    plt.title(f"Shanghai vs Bank Loan Growth", fontdict={'fontsize':20, 'color':'g'})
    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/india_0120.png')



'''
2. 거시경제 지표
2.1 Economic Policy Uncertainty Index for India
'''
def shares_vs_Uncertainty():
    # Economic Policy Uncertainty Index 가 100이상이면 위험으로 간주, 투자 축소
    uncertainty = fred.get_series(series_id='INDEPUINDXM', observation_start=from_date_MT)
    logger2.info(uncertainty[-5:])
    # BSE(Bombay Stock Exchange) + NSE(National Stock Exchange)
    # india_shares = fred.get_series(series_id='SPASTT01INM661N', observation_start=from_date_MT)

    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(india_shares, label='BSE', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(uncertainty, label='Uncertainty', linewidth=1, linestyle='-', color='orange')
    ax2.axhline(y=100, linestyle='--', color='red', linewidth=2)
    plt.title(f"India Shares vs uncertainty", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/india_0210.png')


'''
2.2 Projection of General government gross debt for India
'''
def shares_vs_gov_debt():
    # General government gross debt
    gov_debt = fred.get_series(series_id='GGGDTPINA188N', observation_start=from_date_MT)
    logger2.info(gov_debt[-5:])
    # BSE(Bombay Stock Exchange) + NSE(National Stock Exchange)
    india_shares = fred.get_series(series_id='SPASTT01INM661N', observation_start=from_date_MT)

    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(india_shares, label='BSE', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(gov_debt, label='General government gross debt', linewidth=1, linestyle='-', color='orange')
    ax2.set_ylabel('percentage of GDP')
    plt.title(f"India Shares vs General government gross debt", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/india_0220.png')
    # 20221102 History
    # 2023-01-01    83.837
    # 2024-01-01    84.051
    # 2025-01-01    83.774
    # 2026-01-01    83.395
    # 2027-01-01    83.001








'''
Main Fuction
'''

if __name__ == "__main__":
    cals = eco_indexes(from_date, to_date_2)  # calendars
    india_shares, rupees = bse_vs_inr(from_date_MT, to_date)
    shares_vs_Uncertainty()
    shares_vs_gov_debt()
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