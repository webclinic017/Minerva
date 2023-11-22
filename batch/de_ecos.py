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
to_date_2 = pd.to_datetime(to_date2)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()

# Connect DataBase
database = database_dir+'/'+'Economics.db'
conn, engine = create_connection(database)


'''
0. 경제지표 그래프
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
4. Risk
4.1 [EURO] 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
why 1~2년: 미국 성장이 멈추는 시점(장기 침체, 단기 성장중)은 징후이고, 실제 폭탄은 다른 취약국가에서 외환위기 발생하며 증시폭락
wall street 는 이 시점에 외환위기 발생한 나라에서 12년마다 열리는 대축제를 즐기고 귀한함: 양털깍기
'''
def y10minusm3():
    bond_10y = fred.get_series(series_id='IRLTLT01EZM156N', observation_start=from_date_MT)
    bond_3m = fred.get_series(series_id='IR3TIB01EZM156N', observation_start=from_date_MT)
    bond_10y3m = bond_10y - bond_3m
    bond_us_10y3m = fred.get_series(series_id='T10Y3M', observation_start=from_date_MT)
    crack = 0
    logger2.info('##### 10Y Minus 3M Treasury Constant Maturity #####')
    logger2.info('10Y Minus 3M: \n' + str(bond_10y3m[-16::5]))

    plt.figure(figsize=(19,5))
    plt.title(f"[US vs EURO] 10-Year minus 3-Month Treasury Constant Maturity", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(bond_10y3m, label='[EURO] Resession Indicator after 1~2year: 10y - 3m', linewidth=1, color='maroon')
    plt.plot(bond_us_10y3m, label='[USA] Resession Indicator after 1~2year: 10y - 3m', linewidth=1, color='royalblue')
    plt.legend()
    plt.savefig(reports_dir + '/germany_0400.png')






'''
Main Fuction
'''

if __name__ == "__main__":

    # 1. 경제전망
    cals = eco_calendars(from_date, to_date_2)  # calendars

    # 2. Indicators

    # 3. Markets

    # 4. Risks
    y10minusm3()