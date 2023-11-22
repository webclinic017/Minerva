'''
Prgram 명: Glance of South Korea
Author: jimmy Kang
Mail: jarvisNim@gmail.com
대한민국 경제지표(거시/미시) 부문 엿보기
* investing.com/calendar 포함
History
2022/08/27  Create
2022/09/01  한국은행통계시스템의 최근 데이터로 개선
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''

# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + to_date2)


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
0.1 calendar 정보(from financialmodeling.com) 를 기반으로 그래프 만들기
'''
def eco_calendars(from_date, to_date):

    # 날짜 및 시간 문자열을 날짜로 변환하는 함수
    def parse_date(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()

    # 2) Economics database 에서 쿼리후 시작하는 루틴
    M_table = 'Calendars'
    M_country = 'KR'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        cals = pd.read_sql_query(M_query, conn)
        logger2.info(cals[:30])
    except Exception as e:
        print('Exception: {}'.format(e))

    events = ['Interest Rate Decision', 'PPI MoM ', 'Exports YoY ', 'GDP Growth Rate QoQ ', 'CPI ', 'Industrial Production MoM ', 'Business Confidence ', 'GDP Growth Rate QoQ Adv ', 'Construction Output YoY ', 'Unemployment Rate ', 'Foreign Exchange Reserves ', 'Import Prices YoY ', 'GDP Growth Rate YoY Adv ', 'Retail Sales YoY ', 'Consumer Confidence ', 'Markit Manufacturing PMI ', 'GDP Growth Rate YoY ', 'Current Account ', 'Inflation Rate YoY ', 'Retail Sales MoM ', 'Balance of Trade ', 'Manufacturing Production YoY ', 'S&P Global Manufacturing PMI ', 'Imports YoY ', 'Inflation Rate MoM ', 'Export Prices YoY ', 'Industrial Production YoY ', 'PPI YoY ']

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
    plt.savefig(reports_dir + '/korea_e0000.png')

    return cals


'''
0.2 indicators 정보(from marcovar.com) 를 기반으로 그래프 만들기
'''
def eco_indicators(from_date, to_date):

    # 날짜 및 시간 문자열을 날짜로 변환하는 함수
    def parse_date2(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f").date()

    # 2) Economics database 에서 쿼리후 시작하는 루틴
    M_table = 'Indicators'
    M_country = 'KR'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        indis = pd.read_sql_query(M_query, conn)
        logger2.info(indis[:30])
    except Exception as e:
        print('Exception: {}'.format(e))

    indicators = ['balance of trade', 'bank lending rate', 'business confidence', 'capacity utilization', 'capital flows',\
                  'car registrations', 'central bank balance sheet', 'consumer confidence', 'consumer price index',\
                  'core inflation rate', 'current account', 'current account to gdp', 'deposit interest rate',\
                  'exports', 'external debt', 'fiscal expenditure', 'foreign direct investment', 'foreign exchange reserves',\
                  'gdp', 'Real GDP', 'gdp growth', 'gdp growth annual', 'gold reserves', 'government budget',\
                  'Government budget', 'government debt to gdp', 'government revenues', 'housing index', 'imports', \
                  'industrial production', 'industrial production mom', 'inflation cpi', 'interbank rate',\
                  'interest rate', 'loans to private sector', 'Manufacturing PMI', 'money supply m0', 'money supply m1',\
                  'money supply m2', 'money supply m3', 'Producer Price Index', 'PPI Index', 'retail sales MoM',\
                  'retail sales', 'unemployment rate', 'youth unemployment rate']

    # 전체 그림의 크기를 설정
    plt.figure(figsize=(18, 4*len(indicators)))
    for i, indicator in enumerate(indicators):
        result = indis[indis['Indicator'].str.contains(indicator, case=False, na=False)]
        result['Date'] = result['Date'].apply(parse_date2)
        plt.subplot(len(indicators), 1, i + 1)
        plt.plot(result['Date'], result['Actual'])
        plt.title(indicator)
        plt.xlabel('Date')
        plt.ylabel('Actual')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/korea_e0001.png')

    return indis


'''
1. 통화/금융
1.1 KOSPI 200 vs KRW
'''
def kospi200_vs_krw(from_date, to_date):
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m%d')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m%d')

    # KOSPI
    stat_code  = "802Y001"
    cycle_type = "D"
    item_1 = ['0001000']
    item_2 = []
    item_3 = [ ]

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)

    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')
    kr_total_shares = pd.Series(data=list(df['DATA_VALUE']), index=df['TIME'])

    # USD/KRW
    stat_code  = "731Y003"
    cycle_type = "D"
    item_1 = ['0000003']
    item_2 = []
    item_3 = [ ]

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')
    usd_krw = pd.Series(data=list(df['DATA_VALUE']), index=df['TIME'])

    logger2.info(usd_krw[-26::5])


    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(kr_total_shares, label='KOSPI 200', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(usd_krw, label='USD/KRW', linewidth=1, linestyle='-', color='orange')
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.title(f"KOSPI 200 vs KRW", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/korea_e0100.png')    

    # 세부 확대
    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(kr_total_shares[-72:], label='KOSPI 200', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(usd_krw[-72:], label='USD/KRW', linewidth=1, linestyle='-', color='orange')
    plt.title(f"KOSPI 200 vs KRW", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/korea_e0110.png')

    return kr_total_shares 


'''
1.2 KOSPI 200 vs FED Rates
'''
def kospi200_vs_fred():
    Federal_Rate =  fred.get_series('DFF', observation_start=from_date_MT)
    logger2.info(Federal_Rate[-91::30])

    fig, ax1 = plt.subplots(figsize=(18,4))
    lns1 = ax1.plot(kr_total_shares, label='KOSPI 200', linewidth=1, linestyle='--', color='royalblue')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(Federal_Rate, label='FED Rates', linewidth=1, linestyle='-', color='orange')
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.title(f"KOSPI 200 vs FED Rates", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/korea_e0120.png')
    

'''
1.3 KOSPI 200 vs National Currency
1.3.2 한국은행 기준 M1, M2, M3
'''
def kospi200_vs_currency(from_date, to_date):
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')
    # M1
    stat_code  = "101Y017"
    cycle_type = "M"
    item_1 = ['BBKA00', 'BBKA01']
    item_2 = []
    item_3 = [ ]

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')

    df_m1_month = df.loc[df['ITEM_CODE1'] == 'BBKA00']
    df_m1_cash = df.loc[df['ITEM_CODE1'] == 'BBKA01']

    # M2
    stat_code  = "101Y002"
    cycle_type = "M"
    item_1 = ['BBGA00']
    item_2 = []
    item_3 = [ ]

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')

    df_m2_month = df.loc[df['ITEM_CODE1'] == 'BBGA00']

    # M3
    stat_code  = "112Y002"
    cycle_type = "M"
    item_1 = ['X000000']
    item_2 = []
    item_3 = [ ]

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')

    df_m3_month = df.loc[df['ITEM_CODE1'] == 'X000000']

    logger2.info(df_m2_month[-3:][['TIME','DATA_VALUE']])

    # Graph
    fig, ax = plt.subplots(figsize=(18, 4 * 2))

    # 서브플롯 설정
    plt.subplot(2, 1, 1)
    plt.title(f"M1, M2, M3", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_m1_month['TIME'], df_m1_month['DATA_VALUE'], label='M1 Month End', linewidth=1, color='maroon')
    plt.plot(df_m1_cash['TIME'], df_m1_cash['DATA_VALUE'], label='M1 Cash', linewidth=1, color='yellow')
    plt.plot(df_m2_month['TIME'], df_m2_month['DATA_VALUE'], label='M2 Month End', linewidth=1, color='orange')
    plt.plot(df_m3_month['TIME'], df_m3_month['DATA_VALUE'], label='M3 Month End', linewidth=1, color='green')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f"M1, M2, M3", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_m1_month['TIME'][-37:], df_m1_month['DATA_VALUE'][-37:], label='M1 Month End', linewidth=1, color='maroon')
    # plt.plot(df_m1_cash['TIME'][-37:], df_m1_cash['DATA_VALUE'][-37:], label='M1 Cash', linewidth=1, color='yellow')
    plt.plot(df_m2_month['TIME'][-37:], df_m2_month['DATA_VALUE'][-37:], label='M2 Month End', linewidth=1, color='orange')
    plt.plot(df_m3_month['TIME'][-37:], df_m3_month['DATA_VALUE'][-37:], label='M3 Month End', linewidth=1, color='green')
    plt.legend(loc=3)

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/korea_e0130.png')


'''
1.4 가계대출
'''
def loan(from_date, to_date):
    stat_code  = "151Y005"
    cycle_type = "M"
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')
    item_1 = ['11110A0','11A00A0', '11110B0', '11A00B0']
    item_2 = []
    item_3 = []
    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)

    df_1 = df.loc[df['ITEM_CODE1'] == '11110A0'][['TIME', 'DATA_VALUE', 'UNIT_NAME', 'ITEM_CODE1']]
    df_2 = df.loc[df['ITEM_CODE1'] == '11A00A0'][['TIME', 'DATA_VALUE', 'UNIT_NAME', 'ITEM_CODE1']]
    df_3 = df.loc[df['ITEM_CODE1'] == '11110B0'][['TIME', 'DATA_VALUE', 'UNIT_NAME', 'ITEM_CODE1']]
    df_4 = df.loc[df['ITEM_CODE1'] == '11A00B0'][['TIME', 'DATA_VALUE', 'UNIT_NAME', 'ITEM_CODE1']]

    df_1.set_index('TIME')
    df_2.set_index('TIME')
    df_3.set_index('TIME')
    df_4.set_index('TIME')

    pd.to_datetime(df_1['TIME'], format='%Y%m', errors='coerce').dropna()
    pd.to_datetime(df_2['TIME'], format='%Y%m', errors='coerce').dropna()
    pd.to_datetime(df_3['TIME'], format='%Y%m', errors='coerce').dropna()
    pd.to_datetime(df_4['TIME'], format='%Y%m', errors='coerce').dropna()

    df_1['DATA_VALUE'] = df_1['DATA_VALUE'].astype('float')
    df_2['DATA_VALUE'] = df_2['DATA_VALUE'].astype('float')
    df_3['DATA_VALUE'] = df_3['DATA_VALUE'].astype('float')
    df_4['DATA_VALUE'] = df_4['DATA_VALUE'].astype('float')
    df_1['DATA_VALUE_TOT'] = df_1['DATA_VALUE']+df_2['DATA_VALUE']+df_3['DATA_VALUE']+df_4['DATA_VALUE']

    fig, ax = plt.subplots(figsize=(18, 4 * 2))
    # 서브플롯 설정
    plt.subplot(2, 1, 1)
    plt.title(f"Household Loan", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_1['TIME'], normalize(df_1['DATA_VALUE']), label='House Loan-Bank', linewidth=3)
    plt.plot(df_2['TIME'], normalize(df_2['DATA_VALUE']), label='House Loan-Exclusive Bank', linestyle='--', linewidth=3)
    plt.plot(df_3['TIME'], normalize(df_3['DATA_VALUE']), label='Other Loan-Bank')
    plt.plot(df_4['TIME'], normalize(df_4['DATA_VALUE']), label='Other Loan-Exclusive Bank', linestyle='--')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f"Household Loan", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_1['TIME'][-13:], normalize(df_1['DATA_VALUE'][-13:]), label='House Loan-Bank', linewidth=3)
    plt.plot(df_2['TIME'][-13:], normalize(df_2['DATA_VALUE'][-13:]), label='House Loan-Exclusive Bank', linestyle='--', linewidth=3)
    plt.plot(df_3['TIME'][-13:], normalize(df_3['DATA_VALUE'][-13:]), label='Other Loan-Bank')
    plt.plot(df_4['TIME'][-13:], normalize(df_4['DATA_VALUE'][-13:]), label='Other Loan-Exclusive Bank', linestyle='--')
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/korea_e0140.png')


'''
2. 국민계정
2.1 KOSPI 200(YoY) vs GDP, Industrial Production
'''
def kospi200_vs_gdp_ip(cals, kr_total_shares):
    # GDP (Gross domestic product) : Not Real, just Value
    gdp = cals.loc[cals['event'].str.contains('GDP')]
    # gdp['Date'] = pd.to_datetime(gdp['date'], dayfirst=True)

    gdp_yoy = gdp.loc[gdp['event'].str.contains('YoY')]
    gdp_qoq = gdp.loc[gdp['event'].str.contains('QoQ')]

    # gdp_yoy['Actual/YoY'] = gdp_yoy['actual'].str.rstrip('%').astype('float')
    # gdp_qoq['Actual/QoQ'] = gdp_qoq['actual'].str.rstrip('%').astype('float')
    gdp_yoy['Actual/YoY'] = gdp_yoy['actual']
    gdp_qoq['Actual/QoQ'] = gdp_qoq['actual']

    # Industrial Production
    ip = cals.loc[cals['event'].str.contains('Industrial Production')]
    # ip['Date'] = pd.to_datetime(ip['date'], dayfirst=True)

    ip_yoy = ip.loc[ip['event'].str.contains('YoY')]
    ip_qoq = ip.loc[ip['event'].str.contains('MoM')]

    # ip_yoy['Actual/YoY'] = ip_yoy['actual'].str.rstrip('%').astype('float')
    # ip_qoq['Actual/MoM'] = ip_qoq['actual'].str.rstrip('%').astype('float')
    ip_yoy['Actual/YoY'] = ip_yoy['actual']
    ip_qoq['Actual/MoM'] = ip_qoq['actual']

    logger2.info('GDP (Gross domestic product)')
    logger2.info(gdp_yoy[:4][['Actual/YoY']])

    logger2.info('Industrial Production')
    logger2.info(ip_yoy[:4][['Actual/YoY']])

    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200(YoY) vs GDP, Industrial Production", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(kr_total_shares.pct_change(periods=260)*10, label='KOSPI 200(YoY)', linewidth=1, color='royalblue', linestyle='--')
    plt.plot(gdp_yoy.index, gdp_yoy['Actual/YoY'], label='GDP(YoY)', linewidth=1, \
            linestyle='--', marker='x', markersize=4)
    plt.plot(ip_yoy.index, ip_yoy['Actual/YoY'], label='Industrial Production(YoY)', linewidth=1, \
            linestyle='--')
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0210.png')


'''
2.2 KOSPI 200(MoM) vs GDP, Industrial Production
'''
def kospi200_mom_vs_gdp_ip(cals, kr_total_shares):

    # GDP (Gross domestic product) : Not Real, just Value
    gdp = cals.loc[cals['event'].str.contains('GDP')]
    # gdp['Date'] = pd.to_datetime(gdp['date'], dayfirst=True)

    gdp_yoy = gdp.loc[gdp['event'].str.contains('YoY')]
    gdp_qoq = gdp.loc[gdp['event'].str.contains('QoQ')]

    # gdp_yoy['Actual/YoY'] = gdp_yoy['actual'].str.rstrip('%').astype('float')
    # gdp_qoq['Actual/QoQ'] = gdp_qoq['actual'].str.rstrip('%').astype('float')
    gdp_yoy['Actual/YoY'] = gdp_yoy['actual']
    gdp_qoq['Actual/QoQ'] = gdp_qoq['actual']

    # Industrial Production
    ip = cals.loc[cals['event'].str.contains('Industrial Production')]
    # ip['Date'] = pd.to_datetime(ip['date'], dayfirst=True)

    ip_yoy = ip.loc[ip['event'].str.contains('YoY')]
    ip_qoq = ip.loc[ip['event'].str.contains('MoM')]

    # ip_yoy['Actual/YoY'] = ip_yoy['actual'].str.rstrip('%').astype('float')
    # ip_qoq['Actual/MoM'] = ip_qoq['actual'].str.rstrip('%').astype('float')
    ip_yoy['Actual/YoY'] = ip_yoy['actual']
    ip_qoq['Actual/MoM'] = ip_qoq['actual']

    logger2.info('GDP (Gross domestic product)')
    logger2.info(gdp_yoy[:4][['Actual/YoY']])

    logger2.info('Industrial Production')
    logger2.info(ip_yoy[:4][['Actual/YoY']])


    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200(MoM) vs GDP, Industrial Production", fontdict={'fontsize':20, 'color':'royalblue'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(kr_total_shares.pct_change(periods=21)*10, label='KOSPI 200(YoY)', linewidth=1, linestyle='--')

    plt.plot(gdp_qoq.index, gdp_qoq['Actual/QoQ'], label='GDP(QoQ)', linewidth=1, \
            linestyle='--', marker='x', markersize=4)
    plt.plot(ip_qoq.index, ip_qoq['Actual/MoM'], label='Industrial Production(QoQ)', linewidth=1, \
            linestyle='--')
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0220.png')


'''
2.3 KOSPI 200 vs Real Residential Property Prices
'''
def kospi200_vs_realty(from_date, to_date, kr_total_shares):
    stat_code  = "901Y062"
    cycle_type = "M"
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')
    item_1 = ['P63AD', 'P63AA', 'P63AB', 'P63ACA',]
    item_2 = []
    item_3 = [ ]

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')
            
    df_home_sale_tot = df.loc[df['ITEM_CODE1'] == 'P63AD']
    df_home_sale_house = df.loc[df['ITEM_CODE1'] == 'P63AA']
    df_home_sale_billa = df.loc[df['ITEM_CODE1'] == 'P63AB']
    df_home_sale_apt = df.loc[df['ITEM_CODE1'] == 'P63ACA']

    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200 vs Real Residential Property Prices", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(normalize(kr_total_shares), label='KOSPI 200', linewidth=1, color='royalblue', linestyle='--')
    plt.plot(df_home_sale_tot['TIME'], normalize(df_home_sale_tot['DATA_VALUE']), label='home_sale_tot', linewidth=2, marker='o', markersize=4)
    plt.plot(df_home_sale_tot['TIME'], normalize(df_home_sale_house['DATA_VALUE']), label='home_sale_house', linewidth=2)
    plt.plot(df_home_sale_tot['TIME'], normalize(df_home_sale_billa['DATA_VALUE']), label='Real home_sale_billa', linewidth=1)
    plt.plot(df_home_sale_tot['TIME'], normalize(df_home_sale_apt['DATA_VALUE']), label='home_sale_apt', linewidth=1)
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0230.png')


'''
2.4 실업급여수급실적
'''
def unemployment():
    stat_code  = "901Y084"
    cycle_type = "M"
    start_date = "200601"
    end_date   = "202208"
    item_1 = ['167A']
    item_2 = ['P', 'A']
    item_3 = [ ]
    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3)
    df.drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)

    df_1 = df.loc[df['ITEM_CODE2'] == 'P'][['TIME', 'DATA_VALUE', 'UNIT_NAME', 'ITEM_CODE2']]
    df_2 = df.loc[df['ITEM_CODE2'] == 'A'][['TIME', 'DATA_VALUE', 'UNIT_NAME', 'ITEM_CODE2']]
    df_1.set_index('TIME')
    df_2.set_index('TIME')
    pd.to_datetime(df_1['TIME'], format='%Y%m', errors='coerce').dropna()
    pd.to_datetime(df_2['TIME'], format='%Y%m', errors='coerce').dropna()
    df_1['DATA_VALUE'] = df_1['DATA_VALUE'].astype('int')
    df_2['DATA_VALUE'] = df_2['DATA_VALUE'].astype('int')
    logger2.info(df_1[-3:])

    plt.figure(figsize=(18,6))
    plt.title(f"Unemployement", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_1['TIME'], normalize(df_1['DATA_VALUE']), label='Unemployement')
    plt.plot(df_2['TIME'], normalize(df_2['DATA_VALUE']), label='Unemployement Cost')
    # plt.plot(df_2['TIME'][-10:], df_2['DATA_VALUE'][-10:], label='real_gdp_high', linewidth=1, color='maroon')
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0240.png')


'''
2.5 GDP 디플레이터
GDP 디플레이터
· GDP 디플레이터는 명목 GDP를 실질 GDP로 나누어 사후적으로 계산하는 값이다.
  GDP디플레이터 = (명목 GDP / 실질 GDP) × 100
· 그런데 GDP추계시에는 생산자물가지수(PPI)나 소비자물가지수(CPI) 뿐만 아니라 수출입물가지수, 임금, 환율 등 각종 가격지수가 종합적으로 이용되고 있기 때문에 GDP디플레이터는 국민소득에 영향을 주는 모든 물가요인을 포괄하는 종합적인 물가지수로서 GDP라는 상품의 가격수준을 나타낸다고 할 수 있다.
· 따라서 GDP디플레이터는 생산자물가지수나 소비자물가지수와 함께 국민경제 전체의 물가수준을 나타내는 지표로 사용되기도 한다.
· 한국은행 매분기별 국민소득통계 공표 시 GDP 디플레이터도 포함하여 공표하고 있다.
'''
def deflator():
    stat_code  = "200Y012"
    cycle_type = "Q"

    if cycle_type == "M":
        start_date = datetime.strptime(from_date_MT, '%d/%m/%Y').strftime('%Y%m')
        end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')
    elif cycle_type == "Y":
        start_date = datetime.strptime(from_date_MT, '%d/%m/%Y').strftime('%Y')
        end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y')
    elif cycle_type == "Q":
        start_date = datetime.strptime(from_date_MT, '%d/%m/%Y').strftime('%Y%m')
        start_date = str(start_date[:4]) + 'Q' + str(int(int(start_date[4:6])/3))
        end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')
        end_date = str(end_date[:4]) + 'Q' + str(int(int(end_date[4:6])/3))
    elif cycle_type == "D":
        start_date = datetime.strptime(from_date_MT, '%d/%m/%Y').strftime('%Y%m%d')
        end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m%d')    
    else:
        logger.error("cycle_type Error !!!")
        
    item_1 = ['10101', '10201', '10301', '10401', '10601']
    item_2 = []
    item_3 = []

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    if cycle_type == 'M':    
        df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
        df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    elif (cycle_type == 'Q') or (cycle_type == 'Y'):
        df['TIME'] = df['TIME']
        
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')
    df_consumption = df.loc[df['ITEM_CODE1'] == '10101']# 최종소비지출
    df_gross_capital = df.loc[df['ITEM_CODE1'] == '10201']# 총자본형성
    df_export = df.loc[df['ITEM_CODE1'] == '10301']# 재화와 서비스 수출
    df_import = df.loc[df['ITEM_CODE1'] == '10401']# 재화와 서비스 수입(공제)
    df_gdp_exp = df.loc[df['ITEM_CODE1'] == '10601']# 국내총생산 지출
    df_deflator = df_consumption[['TIME','DATA_VALUE']]+df_gross_capital[['TIME','DATA_VALUE']]+df_export[['TIME','DATA_VALUE']]+df_gdp_exp[['TIME','DATA_VALUE']]+df_import[['TIME','DATA_VALUE']]*(-1)

    logger2.info(df_consumption[['TIME','DATA_VALUE']][-4:]+df_gross_capital[['TIME','DATA_VALUE']][-4:]+df_export[['TIME','DATA_VALUE']][-4:]+df_gdp_exp[['TIME','DATA_VALUE']][-4:]+df_import[['TIME','DATA_VALUE']][-4:]*(-1))

    # Graph
    fig, ax1 = plt.subplots(figsize=(15,5))
    lns1 = ax1.plot(df_consumption['TIME'], (df_consumption['DATA_VALUE']), label='df_consumption', linewidth=1,\
                    linestyle='--')
    lns2 = ax1.plot(df_gross_capital['TIME'], (df_gross_capital['DATA_VALUE']), label='df_gross_capital', linewidth=1,\
                    linestyle='--')
    lns3 = ax1.plot(df_export['TIME'], (df_export['DATA_VALUE']), label='df_export', linewidth=1, linestyle='--')
    lns4 = ax1.plot(df_import['TIME'], (df_import['DATA_VALUE']), label='df_import', linewidth=1, linestyle='--')
    lns5 = ax1.plot(df_gdp_exp['TIME'], (df_gdp_exp['DATA_VALUE']), label='df_gdp_exp', linewidth=1, linestyle='--')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(df_consumption['TIME'], (df_deflator['DATA_VALUE']), label='df_deflator', linewidth=2,\
                    linestyle='-', color='maroon', marker='o', markersize=5)

    plt.title(f"GDP Deflator", fontdict={'fontsize':20, 'color':'g'})
    lns = lns1+lns2+lns3+lns4+lns5
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns6, lns6, loc=1)
    plt.savefig(reports_dir + '/korea_e0250.png')


'''
3. 환율/통관수출입/외환
3.1 KOSPI 200 vs Export/Import/Balance
'''
def kospi200_vs_export_import_balance(cals):
    kospi_yoy = (kr_total_shares - kr_total_shares.shift(260))/kr_total_shares.shift(260)
    # Export
    kor_epi = cals.loc[cals['event'].str.contains('Export Prices')]
    kor_epi['Actual/YoY'] = kor_epi['actual']
    # Import
    kor_ipi = cals.loc[cals['event'].str.contains('Import Prices')]
    kor_ipi['Actual/YoY'] = kor_ipi['actual']
    # Trade Balance
    kor_tb = cals.loc[cals['event'].str.contains('Balance')]
    kor_tb['Actual/Bil/YoY'] = kor_tb['actual']
    logger2.info(kor_tb[kor_tb['Actual/Bil/YoY'] < 0])

    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200 vs Export/Import/Balance", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(normalize(kr_total_shares), label='KOSPI 200', linewidth=1, color='royalblue', linestyle='--')
    # plt.plot(kospi_yoy, label='KOSPI 200/YoY', linewidth=1, color='green', linestyle='--')

    plt.plot(kor_epi.index, normalize(kor_epi['Actual/YoY']), label='Export', linewidth=1, \
            linestyle='--')
    plt.plot(kor_ipi.index, normalize(kor_ipi['Actual/YoY']), label='Import', linewidth=1, \
            linestyle='--')
    plt.plot(kor_tb.index, normalize(kor_tb['Actual/Bil/YoY']), label='Balance', linewidth=1, \
            linestyle='--', marker='x', markersize=4)
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0310.png')

'''
3.2 KOSPI 200 vs Dollar Reserve, Current Account
'''
def kospi200_vs_dollar_current():
    fx_res = cals.loc[cals['event'].str.contains('Foreign Exchange')]
    fx_res['Date'] = pd.to_datetime(fx_res.index, dayfirst=True)
    # fx_res['Actual(Bil)'] = fx_res['actual'].str.rstrip('B').astype('float')
    fx_res['Actual(Bil)'] = fx_res['actual']
    fx_res['Change(Bil)'] = fx_res['Actual(Bil)'] - fx_res['Actual(Bil)'].shift(1)

    cur_account = cals.loc[cals['event'].str.contains('Current Account')]
    cur_account['Date'] = pd.to_datetime(cur_account.index, dayfirst=True)
    # cur_account['Actual(Bil)'] = cur_account['actual'].str.rstrip('B').astype('float')
    cur_account['Actual(Bil)'] = cur_account['actual']

    logger2.info(fx_res[-5:])
    logger2.info(cur_account[cur_account['Actual(Bil)'] < 0])

    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200 vs Dollar Reserve, Current Account", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(normalize(kr_total_shares), label='KOSPI 200', linewidth=1, color='royalblue', linestyle='--')

    plt.plot(fx_res['Date'], normalize(fx_res['Actual(Bil)']), label='Dollar Reserve', linewidth=1, color='maroon',\
            linestyle='--', marker='x', markersize=4)
    plt.plot(cur_account['Date'], normalize(cur_account['Actual(Bil)']), label='Current Account', linewidth=1, \
            color='royalblue', linestyle='--')
    # plt.plot(kor_tb['Date'], normalize(kor_tb['Actual/Bil/YoY']), label='Balance', linewidth=1, color='maroon',\
    #          linestyle='--', marker='x', markersize=4)
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0320.png')


'''
3.3 외환보유액 vs 수출입금액
'''
def dollar_vs_eximport(from_date, to_date):
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')

    # 외환 보유액
    stat_code  = "732Y001"
    cycle_type = "M"
    item_1 = ['99', '04']
    item_2 = []
    item_3 = []

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')
    df_reserve_total = df.loc[df['ITEM_CODE1'] == '99']
    df_reserve_foreign_currency = df.loc[df['ITEM_CODE1'] == '04']
    logger2.info(df_reserve_foreign_currency[['TIME','DATA_VALUE']][-5:])

    # Graph
    plt.figure(figsize=(18,4))
    plt.title(f"Foreign Currency Reserve vs Import_Export", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_reserve_total['TIME'], df_reserve_total['DATA_VALUE'], label='reserve_total', linewidth=1, color='maroon', marker='x')
    plt.plot(df_reserve_foreign_currency['TIME'], df_reserve_foreign_currency['DATA_VALUE'], label='reserve_foreign_currency', linewidth=1, color='green')
    plt.legend()
    plt.savefig(reports_dir + '/korea_e0330.png')


'''
4. 물가 = CPI + PPI + GDP Deflator.Cusumption
4.1 KOSPI 200(YoY) vs PPI(YoY), CPI(YoY)
'''
def kospi200_vs_ppi_cpi(cals, kr_total_shares):
    # PPI: The Producer Price Index (PPI)
    ppi = cals.loc[cals['event'].str.contains('PPI')]
    ppi['Date'] = pd.to_datetime(ppi.index, dayfirst=True)

    ppi_yoy = ppi.loc[ppi['event'].str.contains('YoY')]
    ppi_mom = ppi.loc[ppi['event'].str.contains('MoM')]

    # ppi_yoy['Actual/YoY'] = ppi_yoy['actual'].str.rstrip('%').astype('float')
    ppi_yoy['Actual/YoY'] = ppi_yoy['actual']
    # ppi_mom['Actual/MoM'] = ppi_mom['actual'].str.rstrip('%').astype('float')
    ppi_mom['Actual/MoM'] = ppi_mom['actual']

    # The Consumer Price Index (CPI) 
    cpi = cals.loc[cals['event'].str.contains('Industrial Production')]
    cpi['Date'] = pd.to_datetime(cpi.index, dayfirst=True)

    cpi_yoy = cpi.loc[cpi['event'].str.contains('YoY')]
    cpi_mom = cpi.loc[cpi['event'].str.contains('MoM')]

    # cpi_yoy['Actual/YoY'] = cpi_yoy['actual'].str.rstrip('%').astype('float')
    cpi_yoy['Actual/YoY'] = cpi_yoy['actual']
    # cpi_mom['Actual/YoY'] = cpi_mom['actual'].str.rstrip('%').astype('float')
    cpi_mom['Actual/MoM'] = cpi_mom['actual']

    logger2.info('PPI')
    logger2.info(cpi_yoy.sort_values(by='Date',ascending=False)[:5])
    logger2.info('CPI')
    logger2.info(ppi_yoy.sort_values(by='Date',ascending=False)[:5])

    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200(YoY) vs PPI(YoY), CPI(YoY)", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(kr_total_shares.pct_change(periods=260)*20, label='KOSPI 200(YoY)', linewidth=1, color='royalblue', linestyle='--')
    plt.plot(cpi_yoy['Date'], cpi_yoy['Actual/YoY'], label='CPI(YoY)', linewidth=1, \
            linestyle='--', marker='x', markersize=4)
    plt.plot(ppi_yoy['Date'], ppi_yoy['Actual/YoY'], label='PPI(YoY)', linewidth=1, \
            color='orange', linestyle='--')

    plt.legend()
    plt.savefig(reports_dir + '/korea_e0410.png')

    return ppi_mom, cpi_mom

'''
4.2 KOSPI 200(MoM) vs PPI(MoM), CPI(MoM)
'''
def kospi200_vs_ppim_cpim(kr_total_shares, cpi_mom, ppi_mom):
    plt.figure(figsize=(18,6))
    plt.title(f"KOSPI 200(MoM) vs PPI(MoM), CPI(MoM)", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(kr_total_shares.pct_change(periods=21)*20, label='KOSPI 200(MoM)', linewidth=1, color='royalblue', linestyle='--')
    plt.plot(cpi_mom.index, cpi_mom['Actual/MoM'], label='CPI(MoM)', linewidth=1, \
            linestyle='--', marker='x', markersize=4)
    plt.plot(ppi_mom.index, ppi_mom['Actual/MoM'], label='PPI(MoM)', linewidth=1, \
            color='orange', linestyle='--')

    plt.legend()
    plt.savefig(reports_dir + '/korea_e0420.png')


'''
5. 투자
5.1 증시주변자금 동향
'''
def stock_money_flow(from_date, to_date):
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')

    # 증시주변자금동향
    stat_code  = "901Y056"
    cycle_type = "M"
    item_1 = ['S23A', 'S23B', 'S23E', 'S23F']
    item_2 = []
    item_3 = []

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')

    df_customer_deposits = df.loc[df['ITEM_CODE1'] == 'S23A']# 고객예탁금
    df_deposits_future_option = df.loc[df['ITEM_CODE1'] == 'S23B']# 선물옵션예탁금
    df_money_loans_credit = df.loc[df['ITEM_CODE1'] == 'S23E']# 신용융자잔고
    df_stock_loans_credit = df.loc[df['ITEM_CODE1'] == 'S23F']# 신용대주잔고
    logger2.info(df_customer_deposits[['TIME','DATA_VALUE']][-5:])

    fig, ax = plt.subplots(figsize=(18, 4 * 2))
    # 서브플롯 설정
    plt.subplot(2, 1, 1)
    plt.title(f"Trends of Money in Securities Market", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_customer_deposits['TIME'], normalize(df_customer_deposits['DATA_VALUE']), label='df_customer_deposits', linewidth=2, color='maroon', \
            marker='x', markersize=5)
    plt.plot(df_deposits_future_option['TIME'], normalize(df_deposits_future_option['DATA_VALUE']), label='df_deposits_future_option', linewidth=1)
    plt.plot(df_money_loans_credit['TIME'], normalize(df_money_loans_credit['DATA_VALUE']), label='df_money_loans_credit', linewidth=2)
    plt.plot(df_stock_loans_credit['TIME'], normalize(df_stock_loans_credit['DATA_VALUE']), label='df_stock_loans_credit', linewidth=1)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f"Trends of Money in Securities Market", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_customer_deposits['TIME'][-13:], normalize(df_customer_deposits['DATA_VALUE'][-13:]), label='df_customer_deposits', linewidth=2, color='maroon', \
            marker='x', markersize=5)
    plt.plot(df_deposits_future_option['TIME'][-13:], normalize(df_deposits_future_option['DATA_VALUE'][-13:]), label='df_deposits_future_option', linewidth=1)
    plt.plot(df_money_loans_credit['TIME'][-13:], normalize(df_money_loans_credit['DATA_VALUE'][-13:]), label='df_money_loans_credit', linewidth=2)
    plt.plot(df_stock_loans_credit['TIME'][-13:], normalize(df_stock_loans_credit['DATA_VALUE'][-13:]), label='df_stock_loans_credit', linewidth=1)
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/korea_e0510.png')


'''
5.2 외국인 투자동향
'''
def foreigner_investments(from_date, to_date):
    start_date = datetime.strptime(from_date, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')
    # 외국인 매도/매수/순매수 현황
    stat_code  = "901Y055"
    cycle_type = "M"
    item_1 = ['S22AC', 'S22BC', 'S22CC']
    item_2 = ['VO', 'VA']
    item_3 = []

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')

    df_long_vol = df.loc[df['ITEM_CODE1'] == 'S22AC']#매도/거래량 1
    df_long_vol = df_long_vol.loc[df_long_vol['ITEM_CODE2'] == 'VO']# 매도/거래량 2
    df_short_vol = df.loc[df['ITEM_CODE1'] == 'S22BC']# 매수/거래량 1
    df_short_vol = df_short_vol.loc[df_short_vol['ITEM_CODE2'] == 'VO']# 매수/거래량 2
    df_pure_long_vol = df.loc[df['ITEM_CODE1'] == 'S22CC']# 순매수/거래량 1
    df_pure_long_vol = df_pure_long_vol.loc[df_pure_long_vol['ITEM_CODE2'] == 'VO']# 순매수/거래량 2


    logger2.info(df_pure_long_vol[['TIME','DATA_VALUE']][-5:])

    fig, ax = plt.subplots(figsize=(18, 4 * 2))

    # 서브플롯 설정
    plt.subplot(2, 1, 1)
    plt.title(f"Foreigner's Investments", fontdict={'fontsize':20, 'color':'g'})
    plt.axhline(y=0, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.fill_between(df_pure_long_vol['TIME'], df_pure_long_vol['DATA_VALUE'], y2=0, \
                    where=df_pure_long_vol['DATA_VALUE']<0, color='orange', alpha=0.5)
    plt.grid()
    plt.plot(df_pure_long_vol['TIME'], df_pure_long_vol['DATA_VALUE'], label='df_pure_long_vol', linewidth=2, color='maroon', \
            marker='x', markersize=5)
    # plt.plot(df_long_vol['TIME'], df_long_vol['DATA_VALUE'], label='df_long_vol', linewidth=1)
    # plt.plot(df_short_vol['TIME'], df_short_vol['DATA_VALUE'], label='df_short_vol', linewidth=1)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(f"Foreigner's Investments: Red blocks are BULL.", fontdict={'fontsize':20, 'color':'g'})
    plt.fill_between(df_long_vol['TIME'], y1=df_long_vol['DATA_VALUE'], y2=df_short_vol['DATA_VALUE'], \
                    where=df_long_vol['DATA_VALUE']>df_short_vol['DATA_VALUE'], color='red', alpha=0.9)
    plt.grid()
    plt.plot(df_long_vol['TIME'], df_long_vol['DATA_VALUE'], label='df_long_vol', linewidth=1)
    plt.plot(df_short_vol['TIME'], df_short_vol['DATA_VALUE'], label='df_short_vol', linewidth=1)
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/korea_e0520.png')




'''
Main Fuction
'''

if __name__ == "__main__":
    cals = eco_calendars(from_date, to_date_2)  # calendars
    indis = eco_indicators(from_date, to_date_2) # marcovar
    kr_total_shares = kospi200_vs_krw(from_date_MT, to_date)  # Kospi200
    kospi200_vs_currency(from_date_MT, to_date)
    loan(from_date_MT, to_date)
    kospi200_vs_gdp_ip(cals, kr_total_shares)
    kospi200_mom_vs_gdp_ip(cals, kr_total_shares)
    kospi200_vs_realty(from_date_MT, to_date, kr_total_shares)
    unemployment()
    deflator()
    kospi200_vs_export_import_balance(cals)
    kospi200_vs_dollar_current()
    dollar_vs_eximport(from_date_MT, to_date)
    ppi_mom, cpi_mom = kospi200_vs_ppi_cpi(cals, kr_total_shares)
    kospi200_vs_ppim_cpim(kr_total_shares, cpi_mom, ppi_mom)
    stock_money_flow(from_date_MT, to_date)
    foreigner_investments(from_date_MT, to_date)
