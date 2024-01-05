'''
Prgram 명: Glance of USA
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
미국 투자를 위한 경제지표(거시/미시) 부문 엿보기
History
20231031  Create
20231110  Economics 테이블 쿼리문으로 변경, 기존 url 읽기에서.
20231110  Usa.db 에 미국관련 마켓관련 마스터 와 관련 테이블 구성
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

now = datetime.today()
projection_start = now.strftime('%Y-01-01')
projection_end = (now + relativedelta(years=2)).strftime('%Y-01-01')


# Connect DataBase
database = database_dir+'/'+'Economics.db'
conn, engine = create_connection(database)

def issue_monitoring():
    with conn:
        cur=conn.cursor()
        cur.execute('INSERT INTO Monitors \
                    (Date, Country, Issue, Category, Previous, Estimate, Actual, Change, Impact, ChangePercentage, Bigo) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    ('2023-11-10', 'US', 'Economic Event', 'USD', 100.0, 105.0, 102.5, 2.5, 'High', 2.44, 'Test'))


def sp500_stock(date):
    sp500 = fred.get_series(series_id='SP500', observation_start=date)
    return sp500


'''
1. Summary of Economic Projections (SEP)
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
        logger2.info(cals[-30:])
    except Exception as e:
        logger.error('Exception: {}'.format(e))

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
    plt.savefig(reports_dir + '/us_e0000.png')

    return cals

'''
Key Economic Indicators
'''
# Fed 에서 관심갔고 있는 Key Indacators
def key_indicators():
    lists = []
    # fred 에 트렌드 검색어는.
    from bs4 import BeautifulSoup as bs

    page = requests.get("https://fred.stlouisfed.org/")
    soup = bs(page.text, "html.parser")
    elements = soup.find_all(class_='trending-search-item trending-search-gtm')
    for element in elements:
       lists.append(element.text)
       print(element.text, ' >>>   ', end='')
    print('=====  End =====')
    logger2.info('=== Fed Key Indicatos ===')
    logger2.info(lists)


# Real GDP 추이와 전망
def realGDP_projection():
    # high, low 허들을 넘으면 위험,  central high, central low 허들을 넘으면 경고, 추종은 median
    buf =  fred.get_series('GDPC1', observation_start=from_date_MT)
    real_gdb = round(buf.pct_change(periods=4)*100, 2)
    real_gdp_high = fred.get_series('GDPC1RH', observation_start=projection_start)
    real_gdp_central_high = fred.get_series('GDPC1CTH', observation_start=projection_start)
    real_gdp_median = fred.get_series('GDPC1MD', observation_start=projection_start)
    real_gdp_central_low = fred.get_series('GDPC1CTL', observation_start=projection_start)
    real_gdp_low = fred.get_series('GDPC1RL', observation_start=projection_start)
    logger2.info('##### realGDP_projection #####')
    logger2.info('=== 2023. 11. 10 기준 ===')
    logger2.info('=== 2023-01-01    2.1 ===')
    logger2.info('=== 2024-01-01    1.5 ===')
    logger2.info('=== 2025-01-01    1.8 ===')
    logger2.info('=== 2026-01-01    1.8 ===')
    logger2.info('Real GDP: ' + str(real_gdb[-1]))
    logger2.info('real GDP Project (median): ' + str(real_gdp_median[0]))
    
    plt.figure(figsize=(18,4))
    plt.title(f"Real GDP (Real Gross Domestic Product) Projection", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()

    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)

    plt.plot(real_gdb,  marker='o', label='real_gdp', linewidth=1, markersize=10)
    plt.plot(real_gdp_high, label='real_gdp_high', linewidth=1, color='maroon')
    plt.plot(real_gdp_central_high, label='real_gdp_central_high', linewidth=1, color='chocolate')
    plt.plot(real_gdp_median, label='real_gdp_median', linewidth=3, color='royalblue', marker='x', markersize=9)
    plt.plot(real_gdp_central_low, label='real_gdp_central_low', linewidth=1, color='olive')
    plt.plot(real_gdp_low, label='real_gdp_low', linewidth=1, color='green')
    plt.legend()
    plt.savefig(reports_dir + '/us_e0100.png')


# 실업률 추이와 전망
def unemployment_projection():
    # high, low 허들을 넘으면 위험,  central high, central low 허들을 넘으면 경고, 추종은 median
    Unemployment_Rate =  fred.get_series('UNRATE', observation_start=from_date_MT)
    Unemployment_Rate_high = fred.get_series('UNRATERH', observation_start=projection_start)
    Unemployment_Rate_central_high = fred.get_series('UNRATECTH', observation_start=projection_start)
    Unemployment_Rate_median = fred.get_series('UNRATEMD', observation_start=projection_start)
    Unemployment_Rate_central_low = fred.get_series('UNRATECTL', observation_start=projection_start)
    Unemployment_Rate_low = fred.get_series('UNRATERL', observation_start=projection_start)
    logger2.info('##### unemployment_projection #####')
    logger2.info('=== 2023. 11. 10 기준 ===')
    logger2.info('=== 2023-01-01    3.8 ===')
    logger2.info('=== 2024-01-01    4.1 ===')
    logger2.info('=== 2025-01-01    4.1 ===')
    logger2.info('=== 2026-01-01    4.0 ===')
    logger2.info('Unemployment Rate: ' + str(Unemployment_Rate[-1]))
    logger2.info('Unemployment Rate Median Project (median): ' + str(Unemployment_Rate_median[0]))

    plt.figure(figsize=(15,6))
    plt.title(f"Unemployment Rate", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()

    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(Unemployment_Rate,  marker='o', label='Unemployment_Rate', linewidth=1, markersize=3)
    plt.plot(Unemployment_Rate_high, label='Unemployment_Rate_high', linewidth=1, color='maroon')
    plt.plot(Unemployment_Rate_central_high, label='Unemployment_Rate_central_high', linewidth=1, color='chocolate')
    plt.plot(Unemployment_Rate_median, label='Unemployment_Rate_median', linewidth=3, color='royalblue', marker='x', markersize=9)
    plt.plot(Unemployment_Rate_central_low, label='Unemployment_Rate_central_low', linewidth=1, color='olive')
    plt.plot(Unemployment_Rate_low, label='Unemployment_Rate_low', linewidth=1, color='green')
    plt.legend()
    plt.savefig(reports_dir + '/us_e0200.png')


# PCE 와 Core PCE 추이와 전망
def pce_projection():
    # high, low 허들을 넘으면 위험,  central high, central low 허들을 넘으면 경고, 추종은 median
    buf =  fred.get_series('PCEPI', observation_start=from_date_MT)
    PCE = round(buf.pct_change(periods=12)*100, 2)

    buf =  fred.get_series('PCEPILFE', observation_start=from_date_MT)
    PCE_Core = round(buf.pct_change(periods=12)*100, 2)

    PCE_high = fred.get_series('PCECTPIRH', observation_start=projection_start)
    PCE_central_high = fred.get_series('PCECTPICTH', observation_start=projection_start)
    PCE_median = fred.get_series('PCECTPIMD', observation_start=projection_start)
    PCE_central_low = fred.get_series('PCECTPICTL', observation_start=projection_start)
    PCE_low = fred.get_series('PCECTPIRL', observation_start=projection_start)

    PCE_Core_high = fred.get_series('JCXFERH', observation_start=projection_start)
    PCE_Core_central_high = fred.get_series('JCXFECTH', observation_start=projection_start)
    PCE_Core_median = fred.get_series('JCXFERH', observation_start=projection_start)
    PCE_Core_central_low = fred.get_series('JCXFECTL', observation_start=projection_start)
    PCE_Core_low = fred.get_series('JCXFERL', observation_start=projection_start)

    logger2.info('##### PCE Projection (2023. 11. 10 기준) #####')
    logger2.info('=== 2023-01-01    3.3 ===')
    logger2.info('=== 2024-01-01    2.5 ===')
    logger2.info('=== 2025-01-01    2.2 ===')
    logger2.info('=== 2026-01-01    2.0 ===')
    logger2.info('PCE: ' + str(PCE[-1]))
    logger2.info('PCE Median Project (median): ' + str(PCE_median[0]))

    logger2.info('##### Core PCE Projection (2023. 11. 10 기준) #####')
    logger2.info('=== 2023-01-01    4.2 ===')
    logger2.info('=== 2024-01-01    3.6 ===')
    logger2.info('=== 2025-01-01    3.0 ===')
    logger2.info('=== 2026-01-01    2.9 ===')
    logger2.info('PCE: ' + str(PCE[-1]))
    logger2.info('PCE Median Project (median): ' + str(PCE_median[0]))

    # Graph
    fig, ax = plt.subplots(figsize=(18, 4 * 2))

    # PCE
    plt.subplot(2, 1, 1)
    plt.title(f"PCE (Personal Consumption Expenditures) Inflation", fontdict={'fontsize':20, 'color':'g'})
    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)

    plt.axhline(y=2, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.plot(PCE,  marker='o', label='PCE', linewidth=1, markersize=3)
    plt.plot(PCE_high, label='PCE_high', linewidth=1, color='maroon')
    plt.plot(PCE_central_high, label='PCE_central_high', linewidth=1, color='chocolate')
    plt.plot(PCE_median, label='PCE_median', linewidth=3, color='royalblue', marker='x', markersize=9)
    plt.plot(PCE_central_low, label='PCE_central_low', linewidth=1, color='olive')
    plt.plot(PCE_low, label='PCE_low', linewidth=1, color='green')
    plt.legend()

    # core PCE
    plt.subplot(2, 1, 2)
    plt.title(f"Core PCE (Personal Consumption Expenditures) Inflation  excluding Food, Energy", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()

    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)

    plt.axhline(y=2, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.plot(PCE_Core,  marker='o', linestyle='--', label='PCE_Core', linewidth=1, markersize=3)
    plt.plot(PCE_Core_high, label='PCE_Core_high', linewidth=1, color='maroon', linestyle='--')
    plt.plot(PCE_Core_central_high, label='PCE_Core_central_high', linewidth=1, color='chocolate', linestyle='--')
    plt.plot(PCE_Core_median, label='PCE_Core_median', linewidth=3, color='royalblue', linestyle='--', marker='x', markersize=9)
    plt.plot(PCE_Core_central_low, label='PCE_Core_central_low', linewidth=1, color='olive', linestyle='--')
    plt.plot(PCE_Core_low, label='PCE_Core_low', linewidth=1, color='green', linestyle='--')
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/us_e0300.png')


# Effective Federal Funds Rate 추이와 전망 by 삼프로 KB 연구원...
def effective_rate():
    # high, low 허들을 넘으면 위험,  central high, central low 허들을 넘으면 경고, 추종은 median
    Federal_Rate =  fred.get_series('DFF', observation_start=from_date_MT)
    Federal_Rate_high = fred.get_series('FEDTARRH', observation_start=projection_start)
    Federal_Rate_central_high = fred.get_series('FEDTARCTH', observation_start=projection_start)
    Federal_Rate_median = fred.get_series('FEDTARMD', observation_start=projection_start)
    Federal_Rate_central_low = fred.get_series('FEDTARCTL', observation_start=projection_start)
    Federal_Rate_low = fred.get_series('FEDTARRL', observation_start=projection_start)

    logger2.info('##### Effective Federal Funds Rate Projection (2023. 11. 10 기준) #####')
    logger2.info('=== 2023-01-01    5.6 ===')
    logger2.info('=== 2024-01-01    5.1 ===')
    logger2.info('=== 2025-01-01    3.9 ===')
    logger2.info('=== 2026-01-01    2.9 ===')
    logger2.info('Federal Rate: ' + str(Federal_Rate[-1]))
    logger2.info('Federal_Rate_central Project (low): ' + str(Federal_Rate_central_low[0]))
    logger2.info('Federal_Rate_central Project (high): ' + str(Federal_Rate_central_high[0]))

    plt.figure(figsize=(15,6))
    plt.title(f"Effective Federal Funds Rate", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()

    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(Federal_Rate,  marker='o', label='Federal_Rate', linewidth=1, markersize=2)
    plt.plot(Federal_Rate_high, label='Federal_Rate_high', linewidth=1, color='maroon')
    plt.plot(Federal_Rate_central_high, label='Federal_Rate_central_high', linewidth=1, color='chocolate')
    plt.plot(Federal_Rate_median, label='Federal_Rate_median', linewidth=3, color='royalblue', marker='x', markersize=9)
    plt.plot(Federal_Rate_central_low, label='Federal_Rate_central_low', linewidth=1, color='olive')
    plt.plot(Federal_Rate_low, label='Federal_Rate_low', linewidth=1, color='green')
    plt.legend()
    plt.savefig(reports_dir + '/us_e0400.png')


'''
2. Details
2.1 Leading Indicators OECD (Component series)
BTS - Business situation, CS - Confidence indicator, Construction, Hours, Orders, Interest rate spread, Share prices
'''

def component_series():
    # BTS = BCI (Business confidence index)
    BTS = fred.get_series(series_id='USALOCOBSNOSTSAM', observation_start=from_date_LT)
    # CCI (Consumer confidence index)
    CCI = fred.get_series(series_id='USALOCOCINOSTSAM', observation_start=from_date_LT)
    Construction = fred.get_series(series_id='USALOCOHSNOSTSAM', observation_start=from_date_MT)
    Hours = fred.get_series(series_id='USALOCOHSNOSTSAM', observation_start=from_date_MT)
    Orders = fred.get_series(series_id='USALOCOODNOSTSAM', observation_start=from_date_MT)
    Interest_Rate_Spread = fred.get_series(series_id='USALOCOSINOSTSAM', observation_start=from_date_MT)
    Share_Prices = fred.get_series(series_id='USALOCOSPNOSTSAM', observation_start=from_date_MT)

    logger2.info('##### Leading Indicators OECD #####')
    logger2.info('BTS = BCI (Business confidence index): \n' + str(BTS[-3:]))
    logger2.info('CCI (Consumer confidence index): \n' + str(CCI[-3:]))
    logger2.info('Construction: \n' + str(Construction[-3:]))
    logger2.info('Hours: \n' + str(Hours[-3:]))
    logger2.info('Orders: \n' + str(Orders[-3:]))
    logger2.info('Interest_Rate_Spread: \n' + str(Interest_Rate_Spread[-3:]))
    logger2.info('Share_Prices: \n' + str(Share_Prices[-3:]))   

    plt.figure(figsize=(15,6))
    plt.grid()

    plt.title(f"Leading Indicators OECD: Component series", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3) # Covid-19
    plt.plot(BTS, label='BTS')
    plt.plot(CCI, label='CCI')
    # plt.plot(Construction, label='Construction')
    # plt.plot(Hours, label='Hours')
    # plt.plot(Orders, label='Oders')
    # plt.plot(Interest_Rate_Spread, label='Intrestt Rate Spread')
    plt.plot(Share_Prices, label='Share Prices')
    plt.legend()
    plt.savefig(reports_dir + '/us_e0500.png')


'''
2.2 Nonfarm Payroll YoY Rate
비농업부문 취업율이 Zero(0)로 가면서 폭락이 왔음.
'''
def nonfarm_payroll():
    Nonfarm_Employments = fred.get_series(series_id='PAYEMS', observation_start=from_date_LT)
    df = pd.DataFrame()
    df = Nonfarm_Employments.to_frame(name='Nonfarm_Employments')
    df.reset_index()
    df['Pct_change'] = df['Nonfarm_Employments'].pct_change(periods=12, axis=0)

    huddle_T = df['Pct_change'].loc[df['Pct_change'] > 0].mean() # Top
    print('Huddle TOP: ', huddle_T)
    huddle_B = df['Pct_change'].loc[df['Pct_change'] < 0].mean() # Buttom
    print('Huddle BOTTOM: ',huddle_B)
    print('------------')

    temp = df.iloc[-36:-1] # 직전 3년간
    tp_dates = turning_point(temp, 'Pct_change')
    tp_date = tp_dates.index[0].date()
    print(tp_date)
    trend = trend_detector(temp, 'Pct_change', tp_date)

    logger2.info('##### Nonfarm Payroll YoY Rate #####')
    logger2.info('Nonfarm Payroll YoY Rate: \n' + str(df[-3:]))
    logger2.info('turning point date: \n' + str(tp_dates[-3:]))

    for index, row in temp.iterrows():
        # 트렌드가 하방하면서 huddle_T의 값으로 다가서면서 접근하는지 여부 또는
        if (row['Pct_change'] < huddle_T) and (trend < 0):
            logger2.info(f'트렌드가 하방하면서 huddle_B 의 값으로 다가서면서 접근중: ' + str(row.name))
        # 트레드가 상승하면서 huddle_B의 값으로 다가서면서 접근하는지 여부
        if (row['Pct_change'] > huddle_B) and (trend > 0):
            logger2.info(f'트레드가 상승하면서 huddle_T의 값으로 다가서면서 접근중: ' + str(row.name))


    plt.figure(figsize=(18,4))
    plt.title(f"Nonfarm Payroll YoY Change Rate", fontdict={'fontsize':20, 'color':'g'})
    plt.axhline(y=0, linestyle='-', color='royalblue')
    plt.axhline(y=huddle_T, linestyle='--', color='red', linewidth=1)
    plt.axhline(y=huddle_B, linestyle='--', color='red', linewidth=1)
    plt.axvspan(datetime(1990,6,2), datetime(1991,3,4), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(2001,3,5), datetime(2001,11,5), facecolor='gray', edgecolor='gray', alpha=0.3) # Millaium Crisis
    plt.axvspan(datetime(2007,11,26), datetime(2009,6,1), facecolor='gray', edgecolor='gray', alpha=0.3) # Financial Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.fill_between(df.index, df['Pct_change'], y2=0, color='gray')
    plt.plot(df['Pct_change'])
    plt.legend()
    plt.grid()
    plt.savefig(reports_dir + '/us_e0600.png')


'''
2.3 Nonfarm Hires vs. Nonfarm Job Openings
비농업 구인건수 < 비농업 고용건수 ==> 상승 시작하니 마이크로 분할매수 시작,
비농업 구인건수와 비농업 고용건수의 차이가 최대가 될때까지 상승기 유지.
비농업 구인건수와 비농업 고용건수의 차이가 줄어들기 시작하는 시점(2018년)
그리고 나서 2년뒤 어느날 폭락 또는 침체 갑자기 등장하니 마이크로 분할매도 시작.
'''
def hires_vs_jobopen():
    sp500 = sp500_stock(from_date_LT)
    Nonfarm_Job_Openings = fred.get_series(series_id='JTSJOL', observation_start=from_date_LT)
    Nonfarm_Employments = fred.get_series(series_id='JTSHIL', observation_start=from_date_LT)

    df = pd.DataFrame()
    df = pd.concat([Nonfarm_Employments.rename('Hires'), Nonfarm_Job_Openings.rename('Job_Openings'), sp500.rename('SP500')],\
                   axis=1, join='inner')
    df['Difference'] =  df['Hires'] - df['Job_Openings']
    df['SMA_3'] = df.rolling(3).mean()['Difference']
    df['Slope'] = np.degrees(np.arctan(df['SMA_3'].diff()/3))

    ## 비농업 구인건수 < 비농업 고용건수 ==> 상승 시작하니 마이크로 분할매수 시작
    if len(df.iloc[-11:-1][df['Difference'] > 0]):
        logger2.info('비농업 구인건수 < 비농업 고용건수 ==> 상승 시작하니 마이크로 분할매수 시작')
        logger2.info('nonfarm hire - job open: ' + df.iloc[-11:-1][df['Difference']])
                     
    ## 비농업 구인건수와 비농업 고용건수의 차이가 줄어들기 시작하는 시점(2018년)
    ## 그리고 나서 2년뒤 어느날 폭락 또는 침체 갑자기 등장하니 마이크로 분할매도 시작.
    df['Derivation'] = df['Slope'] - np.roll(df['Slope'], -1) # Use simple difference to compute the derivative
    logger2.info('Derivation: ' + str(df['Derivation'][-4:]))
    logger2.info('>>> Gap to Zero is getting to the Risk.')
    logger2.info('Hires - Job Openings: ')
    logger2.info((df['Hires']-df['Job_Openings'])[-3:])

    plt.figure(figsize=(18,4))
    plt.title(f"Nonfarm Hires vs. Nonfarm Job Openings", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(df['Hires'], label='Hires', linewidth=0.4)
    plt.plot(df['Job_Openings'], label='Job_Openings', linewidth=0.4)
    plt.plot(df['SP500'], label='S&P 500', linewidth=1, color='royalblue')
    plt.fill_between(df.index, y1=df['Hires'], y2=df['Job_Openings'], color='gray', label='Hires-Job_Openings')
    plt.yscale('log')
    plt.legend()
    plt.savefig(reports_dir + '/us_e0700.png')


'''
2.4 S&P500 vs Corp. Profits,   S&P500 vs Real GDP:
미국의 Real GDP 는 분기단위로 가격대 변동추이를 따라고,
한국의 최근 년단위로 가격대 변동추이 따름. 
한국 IMF 사태는 5년단위 가격대 변동추이로 심각한 왜곡이 발생하면서 이를 사전인지하지 못해 IMF 사태를 초래했음.
'''
def sp500_profits_realgdp():
    sp500 = sp500_stock(from_date_MT)
    Corp_Profits = fred.get_series('CP', observation_start=from_date_MT)
    Real_GDP = fred.get_series('GDPC1', observation_start=from_date_MT)

    logger2.info('### Corp. Profits ### \n' + str(Corp_Profits[-3:]))
    logger2.info('### Real GDP ### \n' + str(Real_GDP[-3:]))

    fig, ax1 = plt.subplots(figsize=(15,5))
    lns1 = ax1.plot(sp500, color='royalblue', label='S&P 500')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(Corp_Profits, label='Corp_Profits')
    ax3 = ax2.twinx()
    ins3 = ax3.plot(Real_GDP, label='Real_GDP')
    plt.title(f"S&P500, Corp. Profits vs Real GDP", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns2+ins3
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=6)
    ax2.legend(lns1, lns1, loc=4)
    plt.savefig(reports_dir + '/us_e0800.png')

'''
2.5 Velocity of M2 Money Stock vs Real M2 Money Stock (M2속도 vs M2총량)
'''
def m2v_vs_m2s():
    sp500 = sp500_stock(from_date_MT)
    m2v = fred.get_series(series_id='M2V', observation_start=from_date_MT)
    m2_stock = fred.get_series(series_id='M2REAL', observation_start=from_date_MT)

    logger2.info('M2 total sum: \n' + str(m2_stock[-3:]))
    logger2.info('M2 velocity: \n' + str(m2v[-3:]))

    fig, ax1 = plt.subplots(figsize=(15,5))
    lns1 = ax1.plot(sp500, color='royalblue', label='S&P 500')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(m2v, label='M2 velocity')
    ax3 = ax2.twinx()
    ins3 = ax3.plot(m2_stock, color='orange', label='M2 Total Sum')
    plt.title(f"S&P500, Corp. Profits vs Real GDP", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns2+ins3
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=6)
    ax2.legend(lns1, lns1, loc=4)
    plt.savefig(reports_dir + '/us_e0900.png')


'''
2.5 Unemployment Rate vs Inventories to Sales Ratio(재고/판매 비율)
Unemployment Rate is six months ahead of Inventories to Sales Ratio
실업율이 재고판매비율을 6개월 선행한다.
'''
def unemploy_ahead_retail():
    Retail_Ratio = fred.get_series(series_id='RETAILIRSA', observation_start=from_date_MT)
    Unemployment_Rate = fred.get_series(series_id='UNRATE', observation_start=from_date_MT)

    logger2.info('실업율이 바닥을 찍고 반등하기 시작하는 구간이 매수시점~~~')
    logger2.info('Unemployment Rate: \n' + str(Unemployment_Rate[-3:]))    
    logger2.info('Inventories to Sales Ratio: \n' + str(Retail_Ratio[-3:]))

    plt.figure(figsize=(18,4))
    plt.title(f"Unemployment Rate vs. Reatil Sales Ratio", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(normalize(Retail_Ratio), label='Retail_Sales_Ratio')  # *2 = Scale Rate 
    plt.plot(normalize(Unemployment_Rate), label='Unemployment_Rate')
    plt.legend()
    plt.savefig(reports_dir + '/us_e1000.png')


'''
2.6 5-Year expect inflation vs (10y - 5y) rates+2% Compare
# 10년 기대인플레이션 = 국채 10년물 금리 - 물가연동체 10년물 금리
# 미국의 일반 국채와 인플레이션 연동 미국 국채(TIPS)간 수익률 격차를 나타내는 것으로
# 연방준비제도와 투자자들의 인플레이션 전망치를 가늠할 수 있는 선행지표다. 
# 브레이크 이븐 레이트가 올랐다면 이는 채권 매매자들이 향후 물가상승률이 오를 것이라는 쪽에 베팅을 걸고 있다는 의미로 해석할 수 있다.
'''
def expect_vs_10yminus5y():
    BIR_10y = fred.get_series(series_id='T10YIE', observation_start=from_date_MT)
    BIR_5y = fred.get_series(series_id='T5YIE', observation_start=from_date_MT)
    expect_5y = fred.get_series(series_id='T5YIFR', observation_start=from_date_MT)
    gap = BIR_10y - BIR_5y + 2

    tp = turning_point_for_series(expect_5y)
    tp_date = tp.index[-1] # 직전 TP 일자로 분석
    trend = trend_detector_for_series(BIR_10y, tp_date_from=tp_date)
    
    logger2.info('##### 5-Year expect inflation vs (10y - 5y) rates+2% Compare #####')
    logger2.info('tipping point:  \n' + str(tp[-6:]))
    logger2.info('late tp_date: \n' + str(tp_date))
    logger2.info('trend: \n' + str(trend))
    logger2.info('expect_5y: \n' + str(expect_5y[-51::10]))

    fig, ax = plt.subplots(figsize=(16, 4 * 2))

    # expect_vs_10yminus5y+2%
    plt.subplot(2, 1, 1)
    plt.title(f"10-Year and 5-Year Breakeven Inflation Rate", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(gap, label='gap between 10y bir - 5y bir (Breakeven Inflation Rate)', linewidth=1)
    plt.scatter(tp.index, gap[tp.index], color='red', label='Turning Point') # 특정 일자에 추가적인 점 플로팅
    plt.plot(expect_5y, label='5-Year Forward Inflation Expectation Rate', linewidth=1, color='gray')
    plt.grid()
    plt.legend()

    # Detailed expect_vs_10yminus5y+2%
    plt.subplot(2, 1, 2)
    plt.title(f"10-Year and 5-Year Breakeven Inflation Rate", fontdict={'fontsize':20, 'color':'g'})
    plt.plot(gap[-60:], label='gap between 10y bir - 5y bir (Breakeven Inflation Rate)', linewidth=1)
    plt.scatter(tp_date, gap[tp_date], color='red', label='Turning Point')
    plt.plot(expect_5y[-60:], label='5-Year Forward Inflation Expectation Rate', linewidth=1, color='gray')
    plt.grid()
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/us_e1100.png')


'''
4. Risk Monitoring
url: https://www.federalreserve.gov/econres/notes/feds-notes/predicting-recession-probabilities-using-the-slope-of-the-yield-curve-20180301.html

4.1 Smoothed U.S. Recession Probabilities
Dynamic-factor markov-switching model applied to four monthly coincident variables: 
1) non-farm payroll employment                         2) the index of industrial production
3)real personal income excluding transfer payments     4)and real manufacturing and trade sales.
'''

def smoothed_recession_prob():
    smooth_recession = fred.get_series(series_id='RECPROUSM156N', observation_start=from_date_LT)
    logger2.info('##### Smoothed U.S. Recession Probabilities #####')
    logger2.info('Smoothed Recession Probabilities(%): \n' + str(smooth_recession.loc[smooth_recession > 0.2][-3:]))

    plt.figure(figsize=(16,5))
    plt.title(f"Smoothed U.S. Recession Probabilities", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=1, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.axvspan(datetime(1973,11,15), datetime(1975,3,15), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(1980,2,15), datetime(1980,7,15), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(1981,6,29), datetime(1982,11,1), facecolor='gray', edgecolor='gray', alpha=0.3)# Black Monday
    plt.axvspan(datetime(1990,6,2), datetime(1991,3,4), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(2001,3,5), datetime(2001,11,5), facecolor='gray', edgecolor='gray', alpha=0.3)# Millaium Crisis
    plt.axvspan(datetime(2007,11,26), datetime(2009,6,1), facecolor='gray', edgecolor='gray', alpha=0.3)# Financial Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(smooth_recession, label='Recession Probabilities')
    plt.legend()
    plt.savefig(reports_dir + '/us_e1200.png')

'''
4.2 Real-time Sahm Rule Recession Indicator
when the three-month moving average of the national unemployment rate (U3) rises by 0.50 percentage points
or more relative to its low during the previous 12 months.
'''
def sahm_recession():
    sahm_recession = fred.get_series(series_id='SAHMREALTIME', observation_start=from_date_LT)
    logger2.info('##### Sahm Rule Recession Probabilities #####')
    logger2.info('Sahm Recession Probabilities(%): \n' + str(sahm_recession.loc[sahm_recession > 0.2][-3:]))

    plt.figure(figsize=(19,5))
    plt.title(f"Sahm Rule Recession Indicator", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=0.2, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.axvspan(datetime(1973,11,15), datetime(1975,3,15), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(1980,2,15), datetime(1980,7,15), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(1981,6,29), datetime(1982,11,1), facecolor='gray', edgecolor='gray', alpha=0.3)# Black Monday
    plt.axvspan(datetime(1990,6,2), datetime(1991,3,4), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axvspan(datetime(2001,3,5), datetime(2001,11,5), facecolor='gray', edgecolor='gray', alpha=0.3)# Millaium Crisis
    plt.axvspan(datetime(2007,11,26), datetime(2009,6,1), facecolor='gray', edgecolor='gray', alpha=0.3)# Financial Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(sahm_recession, label='Recession Indicator')
    plt.legend()
    plt.savefig(reports_dir + '/us_e1210.png')

'''
4.3 ICE BofA US High Yield Index Option-Adjusted Spread
'''
def high_yield_spread(CONF_INTVL):
    sp500 = sp500_stock(from_date_MT)
    high_yield_spread = fred.get_series(series_id='BAMLH0A0HYM2', observation_start=from_date_MT)
    crack = high_yield_spread.mean()+ CONF_INTVL*high_yield_spread.std()
    crack_ = high_yield_spread.mean()- CONF_INTVL*high_yield_spread.std()

    logger2.info('##### ICE BofA US High Yield Index Option-Adjusted Spread #####')
    logger2.info('High Yield Spread Crack: \n' + str(high_yield_spread.loc[high_yield_spread > crack][-3:]))

    fig, ax = plt.subplots(figsize=(16, 4 * 2))
    # high_yield_spread
    plt.subplot(2, 1, 1)
    plt.title(f"ICE BofA US High Yield Index Option-Adjusted Spread", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.axhline(y=crack_, linestyle='--', color='red', linewidth=1, label=f"{crack_} % Target Rate")
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(high_yield_spread, label='high_yield_spread', linewidth=1)
    plt.plot(normalize(sp500)*10, label='S&P 500')
    plt.legend()

    # Detailed high_yield_spread
    plt.subplot(2, 1, 2)
    plt.title(f"ICE BofA US High Yield Index Option-Adjusted Spread", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.axhline(y=crack_, linestyle='--', color='red', linewidth=1, label=f"{crack_} % Target Rate")
    plt.plot(high_yield_spread[-24:], label='high_yield_spread')
    plt.plot(normalize(sp500[-24:])*10, label='S&P 500')
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/us_e1300.png')


'''
4.4 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
why 1~2년: 미국 성장이 멈추는 시점(장기 침체, 단기 성장중)은 징후이고, 실제 폭탄은 다른 취약국가에서 외환위기 발생하며 증시폭락
wall street 는 이 시점에 외환위기 발생한 나라에서 10년마다 열리는 대축제를 즐기고 귀한함: 양털깍기
'''
def y10minusy2():
    sp500 = sp500_stock(from_date_MT)
    bond_10y2y = fred.get_series(series_id='T10Y2Y', observation_start=from_date_MT)
    crack = 0

    logger2.info('##### 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity #####')
    logger2.info('10Y Minus 2Y: \n' + str(bond_10y2y[-16::5]))

    plt.figure(figsize=(19,5))
    plt.title(f"10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(bond_10y2y, label='Resession Indicator after 1~2year: 10y - 2y', linewidth=1)
    plt.plot(normalize(sp500), label='S&P 500', color='royalblue')
    plt.legend()
    plt.savefig(reports_dir + '/us_e1400.png')


'''
4.5 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity
why 1~2년: 미국 성장이 멈추는 시점(장기 침체, 단기 성장중)은 징후이고, 실제 폭탄은 다른 취약국가에서 외환위기 발생하며 증시폭락
wall street 는 이 시점에 외환위기 발생한 나라에서 12년마다 열리는 대축제를 즐기고 귀한함: 양털깍기
'''
def y10minusm3():
    bond_us_10y3m = fred.get_series(series_id='T10Y3M', observation_start=from_date_MT)
    tp = turning_point_for_series(bond_us_10y3m)
    tp_date = tp.index[-1]
    trend = trend_detector_for_series(bond_us_10y3m, tp_date_from=tp_date)

    crack = 0
    logger2.info('##### 10Y Minus 3M Treasury Constant Maturity #####')
    logger2.info('10Y Minus 3M: \n' + str(bond_us_10y3m[-16::5]))
    logger2.info('tipping point:  \n' + str(tp[-6:]))
    logger2.info('late tp_date: \n' + str(tp_date))
    logger2.info('trend: \n' + str(trend))

    plt.figure(figsize=(19,5))
    plt.title(f"10Y Minus 3M Treasury Constant Maturity", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(bond_us_10y3m, label='[US] Resession Indicator after 1~2year: 10y - 3m', linewidth=1, color='royalblue')
    plt.legend()
    plt.savefig(reports_dir + '/us_e1500.png')


'''
4.6 Total Assets (Less Eliminations from Consolidation)
'''
def total_assets_and_rp():
    sp500 = sp500_stock(from_date_MT)
    total_assets = fred.get_series(series_id='WALCL', observation_start=from_date_MT)
    rp = fred.get_series(series_id='RRPONTSYD', observation_start=from_date_MT)

    crack = 0

    logger2.info('##### Total Assets (Less Eliminations from Consolidation) #####')
    logger2.info('crack: ' + str(crack))
    logger2.info('Total assets:  \n' + str(total_assets[-5:]))
    logger2.info('rp: \n' + str(rp[-5:]))

    plt.figure(figsize=(18,4))
    plt.title(f"Total Assets (Less Eliminations from Consolidation)", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis

    plt.plot(normalize(total_assets), label='Total Assets')
    plt.plot(normalize(rp.dropna()), label='Overnight Reverse Repurchase Agreements')
    plt.plot(normalize(sp500), label='S&P 500', linewidth=1, color='royalblue')
    plt.legend()
    plt.savefig(reports_dir + '/us_e1600.png')


'''
4.7 10년물 미국채 crack 기준 설정 (분할매수 시작)
why 1~2년: 미국 성장이 멈추는 시점(장기 침체, 단기 성장중)은 징후이고, 실제 폭탄은 다른 취약국가에서 외환위기 발생하며 증시폭락
wall street 는 이 시점에 외환위기 발생한 나라에서 10년마다 열리는 대축제를 즐기고 귀한함: 양털깍기
'''
def y10_crak_line():
    bond_10y = fred.get_series(series_id='DGS10', observation_start=from_date_MT) # 미국채 10년물
    bond_effective = fred.get_series(series_id='FEDFUNDS', observation_start=from_date_MT) # fed effective rate
    crack = bond_effective[-12:].mean() + 0.15   # 15bp 안전마진

    logger2.info('##### 10년물 미국채 crack 기준 설정 (분할매수 시작) #####')
    logger2.info('crack: ' + str(crack))
    logger2.info('10년물 금리:  \n' + str(bond_10y[-5:]))

    plt.figure(figsize=(19,5))
    plt.title(f"Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(bond_10y, label='Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity', linewidth=1)
    plt.legend()
    plt.savefig(reports_dir + '/us_e1700.png')


'''
4.8 rp 자금총량 (Overnight Reverse Repurchase Agreements)
Repurchase Agreement (RP) 또는 Repo 금리:
RP 금리는 은행과 중앙은행 간에 체결된 단기 금융 거래인 리포(Repurchase Agreement)에서 발생하는 이자율을 나타낼 수 있습니다.
리포는 일시적으로 자금이 필요한 은행이 중앙은행에 자산(일반적으로 정부 채권)을 팔아서 현금을 빌리는 거래입니다. 
이자율은 이 거래의 금리를 의미하며 이 자산이 줄어든다는 것은 그만큼 은행들이 보유하고 있는 여유자금이 줄어든다는 뜻이며,
최근 의미로는 은행 리스크가 올라가고 있다는 의미로 해석중임. (2023.04)
'''
def rp_total():
    sp500 = sp500_stock(from_date_MT)
    rp = fred.get_series(series_id='RRPONTSYD', observation_start=from_date_MT)
    crack = rp[-12:].mean() - CONF_INTVL* rp[-12:].std()

    logger2.info('##### rp 자금총량 (Overnight Reverse Repurchase Agreements) #####')
    logger2.info('crack: ' + str(crack))
    logger2.info('rp 자금총량:  \n' + str(rp[-16::5]))

    # Graph
    fig, ax1 = plt.subplots(figsize=(19,5))
    lns1 = ax1.plot(rp, label='Overnight Reverse Repurchase Agreements', linewidth=1,)
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.ylabel('Billions of US Dollars')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(sp500, label='S&P 500', linewidth=1, color='royalblue')
    plt.title(f"Overnight Reverse Repurchase Agreements", fontdict={'fontsize':20, })
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/us_e1800.png')


'''
4.9 Total Bank Reserves
'''
def total_bank_reserves():
    sp500 = sp500_stock(from_date_MT)
    tot_reserves = fred.get_series(series_id='TOTRESNS', observation_start=from_date_MT) # 2month lagging
    reserve_balance = fred.get_series(series_id='WRESBAL', observation_start=from_date_MT) # weekly lagging

    logger2.info('##### Total Bank Reserves #####')
    logger2.info('Total Bank Reserves:  \n' + str(reserve_balance[-16::5]))

    fig, ax1 = plt.subplots(figsize=(19,5))
    lns1 = ax1.plot(tot_reserves, label='Total Reserves', linewidth=1,)
    lns2 = ax1.plot(reserve_balance, label='Reserve Balance', linewidth=2,)
    plt.ylabel('Billions of US Dollars')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(sp500, label='S&P 500', linewidth=1, color='royalblue')
    plt.title(f"Total Bank Reserves (1 year)", fontdict={'fontsize':20, })
    lns = lns1+lns2
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns6, lns6, loc=1)
    plt.savefig(reports_dir + '/us_e1900.png')

'''
4.10 Consumer Loans + Commercial & Industrial Loans, All Commercial Banks
'''
def loans():
    sp500 = sp500_stock(from_date_MT)
    # Consumer Loans, All Commercial Banks
    consumer_loans = fred.get_series(series_id='CONSUMER', observation_start=from_date_MT)
    # Commercial and Industrial Loans, All Commercial Banks
    public_loans = fred.get_series(series_id='TOTCI', observation_start=from_date_MT)

    logger2.info('##### Consumer Loans + Commercial & Industrial Loans, All Commercial Banks #####')
    logger2.info('Consumer Loans:  \n' + str(consumer_loans[-16::5]))
    logger2.info('Commercial & Industrial Loans:  \n' + str(public_loans[-16::5]))

    # Graph
    fig, ax1 = plt.subplots(figsize=(19,5))
    lns1 = ax1.plot(consumer_loans, label='Consumer Loans', linewidth=1, marker='x')
    lns2 = ax1.plot(public_loans, label='Commercial and Industrial Loans', linewidth=1,linestyle='--', marker='x')
    plt.ylabel('Billions of US Dollars')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(sp500, label='S&P 500', linewidth=1, color='green')
    plt.title(f"Consumer Loans, All Commercial Banks", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1+lns2
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns6, lns6, loc=1)
    plt.savefig(reports_dir + '/us_e2000.png')

'''
4.11 언제 지원금이 다 소진되는가 ? feat. Savings.
코로나 기간 적립한 저축액에 대비 남은 금액을 다 소진하게되는 시간은 ?
'''
def when_is_the_day():

    # Personal Saving
    personal_saving = fred.get_series(series_id='PMSAVE', observation_start='01/03/2020')
    # Personal Saving Rate
    personal_saving_rate = fred.get_series(series_id='PSAVERT', observation_start='01/03/2020')

    crack = personal_saving['2020-01-01':'2020-03-01'].mean()  # 코로나이전 월평균 저축액

    _extra_saving=0
    for i, x in enumerate(personal_saving.index):
        if personal_saving[x] > crack:
            _extra_saving += personal_saving[x] - crack   # 초과저축액 찾기
    logger2.info('##### when_is_the_day ? #####')
    logger2.info('총 초과저축액:  ' + str(_extra_saving))

    for i,x in enumerate(personal_saving.index): # crack 을 계속 저축한다는 전제로, 그 보다 낮은 금액은 초과액에서 차감하는 방식
        if personal_saving[x] <= crack:
            _extra_saving -= (crack - personal_saving[x])
            logger2.info('평균저축금액 이하만큼 월 차감액:  ' + str(i) + '개월   ' + str(round(_extra_saving,0)) + ' Billion dollars 남아있음.')
    
    logger2.info('#######################################')
    logger2.info('남은 예상개월수: ' + str(round(_extra_saving / personal_saving[-1])))     

    # Graph
    fig, ax1 = plt.subplots(figsize=(19,5))
    lns1 = ax1.plot(personal_saving, label='Personal Saving', linewidth=1,linestyle='--')
    plt.axhline(y=crack, linestyle='--', color='red', linewidth=1, label=f"{crack} % Target Rate")
    plt.ylabel('Billions of US Dollars')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(personal_saving_rate, label='Personal Saving Rate', linewidth=1, color='green')
    plt.ylabel('Percent')
    plt.title(f"Savings", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/us_e2100.png')



'''
Main Fuction
'''

if __name__ == "__main__":

    # 1. Projections: Summary of Economic Projections (SEP)
    cals = eco_calendars(from_date, to_date_2)  # calendars
    key_indicators()
    realGDP_projection()
    unemployment_projection()
    pce_projection()
    effective_rate()

    # 2. Indicators
    component_series()
    nonfarm_payroll()
    hires_vs_jobopen()
    sp500_profits_realgdp()
    m2v_vs_m2s()
    unemploy_ahead_retail()
    expect_vs_10yminus5y()

    
    # 3. Markets


    # 4. Risks
    smoothed_recession_prob()
    sahm_recession()
    high_yield_spread(CONF_INTVL)
    y10minusy2()
    y10minusm3()
    total_assets_and_rp()
    y10_crak_line()
    rp_total()
    total_bank_reserves()
    loans()
    when_is_the_day()