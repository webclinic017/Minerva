'''
Prgram 명: Glance of India
Author: jimmy Kang
Mail: jarvisNim@gmail.com
일본 투자를 위한 경제지표(거시/미시) 부문 엿보기
*인도는 과연 중국을 이을 다음 글로벌 공장이 될수 있을 것인가 ?
History
20221101  Create
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
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
_to_date = pd.to_datetime(to_date)
three_month_days = relativedelta(weeks=12)
from_date = (_to_date - three_month_days).date()
to_date = _to_date.date()

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
    M_country = 'IN'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        cals = pd.read_sql_query(M_query, conn)
        logger2.info('=== Calendars ===')
        logger2.info(cals[-30:])
    except Exception as e:
        print('Exception: {}'.format(e))

    events = ['Eco Watchers Survey Current ', 'GDP Price Index YoY ', 'GDP External Demand QoQ ', 'Exports YoY ', 'Eco Watchers Survey Outlook ', 'Capital Spending YoY ', 'BSI Large Manufacturing QoQ ', 'Leading Economic Index ', 'Average Cash Earnings YoY ', 'GDP Growth Annualized ', 'Tokyo Core CPI YoY ', 'Coincident Index ', 'Tertiary Industry Index MoM ', 'Inflation Rate MoM ', 'Tankan Large Manufacturing Outlook ', 'Jobs/applications ratio ', 'Stock Investment by Foreigners ', 'Inflation Rate Ex-Food and Energy YoY ', 'Housing Starts YoY ', 'Household Spending YoY ', 'Core Inflation Rate YoY ', 'Tankan Large All Industry Capex ', 'PPI MoM ', 'Overtime Pay YoY ', 'Machinery Orders YoY ', 'Jibun Bank Services PMI ', 'Tankan Large Non-Manufacturing Index ', 'Current Account ', 'BoJ Interest Rate Decision', 'Unemployment Rate ', 'Industrial Production MoM ', 'Reuters Tankan Index ', 'Imports YoY ', '5 Year Note Yield', 'Foreign Exchange Reserves ', '5-Year JGB Auction', '52-Week Bill Auction', 'Tankan Non-Manufacturing Outlook ', 'GDP Growth Rate QoQ ', 'Machinery Orders MoM ', 'Foreign Bond Investment ', 'Household Spending MoM ', 'Jibun Bank Manufacturing PMI ', 'Industrial Production MoM. ', 'GDP Capital Expenditure QoQ ', 'Balance of Trade ', '6-Month Bill Auction', 'Jibun Bank Composite PMI ', '40-Year JGB Auction', '5-Year Note Yield', 'Industrial Production YoY ', 'PPI YoY ', 'Retail Sales YoY ', 'Inflation Rate YoY ', '30-Year JGB Auction', 'Tankan Large Manufacturers Index ', 'Tokyo CPI YoY ', 'Tankan Small Manufacturers Index ', '2 Year Note Yield', 'Bank Lending YoY ', 'Consumer Confidence ', '2-Year JGB Auction', '3-Month Bill Auction', 'Machine Tool Orders YoY ', 'BoJ Nakamura Spech', 'GDP Private Consumption QoQ ', 'Retail Sales MoM ', 'Tokyo CPI Ex Food and Energy YoY ', 'Construction Orders YoY ', 'Capacity Utilization MoM ']

    # 전체 그림의 크기를 설정
    plt.figure(figsize=(16, 4*len(events)))
    for i, event in enumerate(events):
        result = cals[cals['event'].str.contains(event, case=False, na=False)]
        if result.empty:
            continue
        result['date'] = result['date'].apply(parse_date)
        plt.subplot(len(events), 1, i + 1)
        plt.plot(result['date'], result['actual'])
        max_val = max(result['actual'])
        min_val = min(result['actual'])
        if (max_val > 0) and (min_val < 0):       # 시각효과     
            plt.axhline(y=0, linestyle='--', color='red', linewidth=1)            
        plt.title(event)
        plt.grid()
        plt.xlabel('date')
        plt.ylabel('actual')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/in_e0000.png')

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
    plt.savefig(reports_dir + '/in_e0110.png')

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
    plt.savefig(reports_dir + '/in_e0111.png')

    return india_shares, rupees


'''
1.2 Shares vs PBoC Loan Prime Rate
'''
def shares_vs_loan(cals):
    # Bank Loan Growth
    buf = cals.loc[cals['event'].str.contains('Bank Loan Growth')]
    logger2.info('=== India Loan ===')
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
    plt.savefig(reports_dir + '/in_e0120.png')



'''
2. 거시경제 지표
2.1 Economic Policy Uncertainty Index for India
'''
def shares_vs_Uncertainty():
    # Economic Policy Uncertainty Index 가 100이상이면 위험으로 간주, 투자 축소
    uncertainty = fred.get_series(series_id='INDEPUINDXM', observation_start=from_date_MT)
    logger2.info('=== India Uncertainty ===')
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
    plt.savefig(reports_dir + '/in_e0210.png')


'''
2.2 Projection of General government gross debt for India
'''
def shares_vs_gov_debt():
    # General government gross debt
    gov_debt = fred.get_series(series_id='GGGDTPINA188N', observation_start=from_date_MT)
    logger2.info('=== India Government Debt ===')
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
    plt.savefig(reports_dir + '/in_e0220.png')
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
    cals = eco_calendars(from_date, to_date)  # calendars
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