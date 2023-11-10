'''
Prgram 명: Glance of USA
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
    M_country = 'US'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        cals = pd.read_sql_query(M_query, conn)
        logger2.info(cals[:30])
    except Exception as e:
        print('Exception: {}'.format(e))

    events = ['EIA Distillate Fuel Production Change ', 'Retail Sales Ex Autos MoM ', \
            'U-6 Unemployment Rate ', 'Real Consumer Spending QoQ ', '10-Year TIPS Auction', \
            'Interest Rate Projection - 1st Yr', 'Employment Cost Index QoQ ', \
            'Industrial Production YoY ', 'GDP Growth Rate QoQ Adv ', 'Dallas Fed Manufacturing Index ', \
            'Philly Fed Employment ', 'CB Consumer Confidence ', 'Imports ', 'NFIB Business Optimism Index ', \
            'Manufacturing Payrolls ', 'Pending Home Sales MoM ', 'Core PCE Prices QoQ Adv ', \
            'International Monetary Market ', 'Initial Jobless Claims ', 'Michigan Inflation Expectations ', \
            'Retail Sales MoM ', 'Goods Trade Balance Adv ', 'Financial Accounts of the United States', \
            'Chicago Fed National Activity Index ', 'Retail Sales Ex Gas/Autos MoM ', \
            'ISM Manufacturing PMI ', 'CB Leading Index MoM ', 'S&P Global Manufacturing PMI ', \
            'Michigan Consumer Expectations ', 'Goods Trade Balance ', '3-Month Bill Auction', 'Core Inflation Rate YoY ', \
            'Nonfarm Productivity QoQ ', \
            'S&P/Case-Shiller Home Price MoM ', 'Employment Cost - Benefits QoQ ', 'PCE Price Index MoM ', \
            'ISM Services Prices ', 'Kansas Fed Manufacturing Index ', 'Unit Labour Costs QoQ ', '3-Year Note Auction', \
            'Personal Income MoM ', 'Building Permits ', 'Corporate Profits QoQ ', '7-Year Note Auction', \
            'Richmond Fed Services Index ', 'Retail Sales YoY ', 'Average Hourly Earnings YoY ', 'API Crude Oil Stock Change ', \
            'Jobless Claims 4-Week Average ', 'Pending Home Sales YoY ', 'Retail Inventories Ex Autos MoM ', \
            '52-Week Bill Auction', 'PPI Ex Food, Energy and Trade YoY ', \
            'Richmond Fed Manufacturing Shipments Index ', 'CPI ', 'Michigan Consumer Sentiment ', \
            'EIA Crude Oil Imports Change ', 'PPI YoY ', 'Average Hourly Earnings MoM ', 'NAHB Housing Market Index ', \
            'Non Farm Payrolls ', 'EIA Crude Oil Stocks Change ', 'Inflation Rate MoM ', 'Unemployment Rate ', \
            'Kansas Fed Composite Index ', 'Wholesale Inventories MoM ', 'New Home Sales MoM ', \
            'GDP Price Index QoQ ', 'Redbook YoY ', 'Fed Interest Rate Decision', 'ISM Services Employment ', \
            'LMI Logistics Managers Index Current ', '10-Year Note Auction', 'Import Prices MoM ', 'Average Weekly Hours ', \
            'Michigan 5 Year Inflation Expectations ', '8-Week Bill Auction', 'Manufacturing Production YoY ', \
            'Manufacturing Production MoM ', 'NY Empire State Manufacturing Index ', 'Import Prices YoY ', \
            'Factory Orders ex Transportation ', 'Net Long-Term TIC Flows ', \
            'PCE Prices QoQ Adv ', 'Michigan Current Conditions ', \
            '4-Week Bill Auction', 'Nonfarm Payrolls Private ', 'PCE Prices QoQ ', 'Core PCE Price Index YoY ', \
            'S&P/Case-Shiller Home Price YoY ', 'Philly Fed CAPEX Index ', 'Existing Home Sales MoM ', \
            'Interest Rate Projection - Longer', 'Balance of Trade ', 'Dallas Fed Services Revenues Index ', \
            'New Home Sales ', 'PPI Ex Food, Energy and Trade MoM ', 'JOLTs Job Quits ', '30-Year TIPS Auction', \
            'Total Household Debt ', 'Durable Goods Orders Ex Transp MoM ', 'Employment Cost - Wages QoQ ', \
            '2-Year Note Auction', 'EIA Gasoline Stocks Change ', 'IBD/TIPP Economic Optimism ', 'Personal Spending MoM ', \
            'Core PPI MoM ', 'EIA Gasoline Production Change ', 'Used Car Prices MoM ', 'Net Long-term TIC Flows ', \
            'Philadelphia Fed Manufacturing Index ', 'GDP Sales QoQ ', 'Used Car Prices YoY ', 'Export Prices YoY ', \
            'Interest Rate Projection - 2nd Yr', 'MBA Mortgage Applications ', '6-Month Bill Auction', '5-Year Note Auction', \
            'Interest Rate Projection - Current', 'Non Defense Goods Orders Ex Air ', 'EIA Distillate Stocks Change ', \
            'Philly Fed Business Conditions ', '30-Year Bond Auction', 'ISM Services PMI ', 'LMI Logistics Managers Index Current', \
            'PCE Price Index YoY ', 'Real Consumer Spending QoQ Adv ', 'Jobless Claims 4-week Average ', '2-Year FRN Auction', \
            'S&P Global Composite PMI ', 'Consumer Inflation Expectations ',  \
            'Foreign Bond Investment ', 'Participation Rate ', 'ISM Manufacturing Employment ', 'Building Permits MoM ', \
            '20-Year Bond Auction', 'Housing Starts MoM ', 'EIA Cushing Crude Oil Stocks Change ', 'House Price Index ', \
            'Industrial Production MoM ', 'Core Inflation Rate MoM ', 'Inflation Rate YoY ', 'ADP Employment Change ', \
            'Construction Spending MoM ', 'FOMC Economic Projections', 'GDP Growth Rate QoQ ', 'Overall Net Capital Flows ', \
            'Factory Orders MoM ', 'Core PCE Price Index MoM ', 'ISM Services New Orders ', \
            'ISM Services Business Activity ', 'S&P Global Services PMI ', 'MBA Mortgage Refinance Index ', \
            'Business Inventories MoM ', 'Core PPI YoY ', 'EIA Heating Oil Stocks Change ', 'GDP Price Index QoQ Adv ', \
            '30-Year Mortgage Rate ', '5-Year TIPS Auction', 'Government Payrolls ', 'ISM Manufacturing Prices ', \
            'Chicago PMI ', 'House Price Index MoM ', 'CPI s.a ', 'House Price Index YoY ', \
            'Baker Hughes Oil Rig Count ', 'JOLTs Job Openings ', 'Exports ', 'Retail Inventories Ex Autos MoM Adv ', \
            'Dallas Fed Services Index ', 'Core PCE Prices QoQ ', 'Durable Goods Orders ex Defense MoM ', \
            'Philly Fed Prices Paid ', '15-Year Mortgage Rate ', 'ISM Manufacturing New Orders ', \
            'Continuing Jobless Claims ', 'Consumer Credit Change ', 'PPI MoM ', 'Interest Rate Projection - 3rd Yr', \
            'MBA Mortgage Market Index ', 'Personal Income ', '17-Week Bill Auction', 'Existing Home Sales ', \
            'MBA 30-Year Mortgage Rate ', 'Capacity Utilization ', 'GDP Sales QoQ Adv ',  \
            'Export Prices MoM ', 'EIA Natural Gas Stocks Change ', \
            'Richmond Fed Manufacturing Index ', 'Challenger Job Cuts ', 'Current Account ', 'Durable Goods Orders MoM ', \
            'Philly Fed New Orders ', 'Wholesale Inventories MoM Adv ', \
            'MBA Purchase Index ', 'Baker Hughes Total Rig Count ', 'EIA Refinery Crude Runs Change ', 'Total Vehicle Sales ', \
            'Housing Starts ']
    
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
    plt.savefig(reports_dir + '/usa_0000.png')

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