'''
Prgram 명: Glance of China
Author: jimmy Kang
Mail: jarvisNim@gmail.com
중국 투자를 위한 경제지표(거시/미시) 부문 엿보기
* investing.com/calendar 포함
History
20220907  Create
20221116  Calendars.db 로부터 지표정보 읽어오기(FreeStockMarket API 로 전환)
20231020  financialmodeling API 로 다시 전환
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
    M_country = 'CN'
    M_query = f"SELECT * from {M_table} WHERE country = '{M_country}'"

    try:
        cals = pd.read_sql_query(M_query, conn)
        logger2.info(cals[:30])
    except Exception as e:
        print('Exception: {}'.format(e))

    events = ['Loan Prime Rate 5Y', 'Caixin Composite PMI ', 'GDP Growth Rate QoQ ', 'New Yuan Loans ', 'Caixin Services PMI ', \
            'M2 Money Supply YoY ', 'Outstanding Loan Growth YoY ', 'NBS General PMI ', 'Caixin Manufacturing PMI ', 'Loan Prime Rate 5Y ', \
            'Industrial Profits YoY ', 'Total Social Financing ', 'Industrial Production YoY ', 'Industrial Capacity Utilization ', \
            'NBS Manufacturing PMI ', 'NBS Non Manufacturing PMI ', 'Unemployment Rate ', 'Foreign Exchange Reserves ', \
            'House Price Index YoY ', 'Imports YoY ', 'Current Account ', 'Exports YoY ', 'Balance of Trade ', 'PPI YoY ', 'FDI ', \
            'Inflation Rate MoM ', 'Inflation Rate YoY ', 'PBoC 1-Year MLF Announcement', 'Fixed Asset Investment ', 'Loan Prime Rate 1Y', \
            'Retail Sales YoY ', 'GDP Growth Rate YoY ', 'Vehicle Sales YoY ', 'Industrial Profits ']

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
    plt.savefig(reports_dir + '/cn_e0000.png')

    return cals

'''
1. 통화/금융
1.1 상하이, 심천 vs CNY
'''
def shanghai_szse_vs_yuan(from_date, to_date):
    # investpy 403 Error
    # shanghai_shares = inv.get_index_historical_data(index='Shanghai', country='china', from_date=from_date_MT, to_date=to_date)['Close']
    # szse_shares = inv.get_index_historical_data(index='SZSE Component', country='china', from_date=from_date_MT, to_date=to_date)['Close']
    # usd_cny = inv.get_currency_cross_historical_data(currency_cross='USD/CNY', from_date=from_date_MT, to_date=to_date)
    # print(usd_cny[-26::5])

    # Alternative

    # shanghai_shares
    ticker = yf.Ticker("000001.SS")
    _from_date_MT = pd.to_datetime(from_date_MT).strftime('%Y-%m-%d')
    shanghai_shares = ticker.history(start=_from_date_MT)

    # szse_shares
    ticker = yf.Ticker("399107.SZ")
    szse_shares = ticker.history(start=_from_date_MT)

    # Chinese Yuan Renminbi
    yuan = fred.get_series(series_id='DEXCHUS', observation_start=from_date_MT)

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(shanghai_shares.index, shanghai_shares['Close'], label='Shanghai', linewidth=1, linestyle='--')
    lns2 = ax1.plot(szse_shares.index, szse_shares['Close'], label='SZSE', linewidth=1, linestyle='--')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(yuan, label='USD/CNY', linewidth=1, linestyle='-', color='red')

    plt.title(f"China Shares vs CNY", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1+lns2
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=4)
    plt.savefig(reports_dir + '/cn_e0110.png')

    return shanghai_shares, szse_shares

'''
1.2 Shanghai vs PBoC Loan Prime Rate
'''
def shanghai_vs_loan(cals):
    buf = cals.loc[cals['event'].str.contains('Loan Prime Rate 1Y')]
    buf['Date'] = pd.to_datetime(buf.date).dt.date
    buf['Actual'] = buf['actual'].astype('float')
    buf = buf.dropna(subset=['Actual'])
    buf['Date'].reset_index()
    logger2.info(f"Shanghai vs PBoC Loan Prime Rate".center(60, '*'))
    logger2.info(buf[:5])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(buf['Date'], buf['Actual'], label='PBoC Loan Prime Rate', linewidth=2,\
                    linestyle='-', marker='x', color='maroon')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], label='Shanghai', linewidth=1, linestyle='-')
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.title(f"Shanghai vs PBoC Loan Prime Rate", fontdict={'fontsize':20, 'color':'g'})
    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0120.png')


'''
1.3 Shanghai vs vs National Currency: M2(YoY)
'''
def shanghai_vs_m2(cals):
    buf = cals.loc[cals['event'].str.contains('M2')]
    buf['Date'] = pd.to_datetime(buf.date).dt.date
    buf['Actual'] = buf['actual'].astype('float')
    buf = buf.dropna(subset=['Actual'])
    buf['Date'].reset_index()    
    logger2.info(buf[:10:2])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(buf['Date'], buf['Actual'], label='M2(YoY)', linewidth=2,\
                    linestyle='-', marker='x', color='maroon')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], label='Shanghai', linewidth=1, linestyle='-')
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.title(f"Shanghai vs M2(YoY)", fontdict={'fontsize':20, 'color':'g'})
    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=4)
    plt.savefig(reports_dir + '/cn_e0130.png')


'''
** 1.4 가계대출
'''
def house_loan(cals):
    # New Loans
    new_loans = cals.loc[cals['event'].str.contains('New Yuan Loans')]
    new_loans['Date'] = pd.to_datetime(new_loans.date).dt.date
    new_loans['Actual'] = new_loans['actual'].astype('float')
    new_loans = new_loans.dropna(subset=['Actual'])
    new_loans['Date'].reset_index()      
    logger2.info(new_loans[:10:2])

    # Outstanding Loan Growth (미결제대출 증가율)
    loan_growth = cals.loc[cals['event'].str.contains('Outstanding Loan Growth YoY')]
    loan_growth['Date'] = pd.to_datetime(loan_growth.date).dt.date
    loan_growth['Actual'] = loan_growth['actual'].astype('float')
    loan_growth = loan_growth.dropna(subset=['Actual'])
    loan_growth['Date'].reset_index()   
    logger2.info(loan_growth[:10:2])


    # Graph: New Loans vs Outstanding Loan Growth (미결제대출 증가율)
    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(new_loans['Date'], new_loans['Actual'], label='New Loans', linewidth=1,\
                    linestyle='-', color='maroon')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(loan_growth['Date'], loan_growth['Actual'], label='Outstanding Loan Growth', linewidth=1,\
                    linestyle='-', color='royalblue')
    plt.title(f"New Loans vs Outstanding Loan Growth", fontdict={'fontsize':20, 'color':'g'})
    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0140.png')

    logger2.info('***************************************************************')
    logger2.info('월단위로 미결제 대출이 증가를 시작하면 항상 신규대출이 늘었으며, ')
    logger2.info('모든 경우 미결제대출이 증가를 시작하면 이후 5개월은 크게 하방압력을 행사함.')
    logger2.info('이는 정부가 미결제 대출이 늘어나는 징후만 보이면 상하이 주가는 장기간 크게 하락함.')
    logger2.info('***************************************************************')

    # Graph: Shanghai vs Outstanding Loan Growth (미결제대출 증가율)
    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(loan_growth['Date'], loan_growth['Actual'], label='New Loans', linewidth=1,\
                    linestyle='-', color='royalblue')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], label='Shanghai', linewidth=1, linestyle='-', color='green')
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.title(f"** Shanghai vs Outstanding Loan Growth", fontdict={'fontsize':20, 'color':'g'})
    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0141.png')


'''
1.5 Chinese Yuan Renminbi to U.S. Dollar Spot Exchange Rate
'''
def yuan_exchange_rate(from_date):
    # Chinese Yuan Renminbi
    yuan = fred.get_series(series_id='DEXCHUS', observation_start=from_date)

    # Graph: Shanghai vs GDP (Gross domestic product)
    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(yuan, label='Chinese Yuan Renminbi', linewidth=1,)
    ax2 = ax1.twinx()
    lns4 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)

    plt.title(f"Chinese Yuan Renminbi to U.S. Dollar Spot Exchange Rate", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns4, lns4, loc=1)
    plt.savefig(reports_dir + '/cn_e0150.png')


'''
2. 국민계정
2.1 Shanghai vs GDP(YoY, QoQ, YTD)
'''
def shanghai_vs_gpd(cals):
    gdp_yoy = cals.loc[cals['event'].str.contains('GDP Growth Rate YoY', regex=False)]
    gdp_yoy['Date'] = pd.to_datetime(gdp_yoy.date).dt.date
    gdp_yoy['Actual/YoY'] = gdp_yoy['actual'].astype('float')
    gdp_yoy = gdp_yoy.dropna(subset=['Actual/YoY'])
    gdp_yoy['Date'].reset_index()     

    gdp_qoq = cals.loc[cals['event'].str.contains('GDP Growth Rate QoQ', regex=False)]
    gdp_qoq['Date'] = pd.to_datetime(gdp_qoq.date).dt.date
    gdp_qoq['Actual/QoQ'] = gdp_qoq['actual'].astype('float')
    gdp_qoq = gdp_qoq.dropna(subset=['Actual/QoQ'])
    gdp_qoq['Date'].reset_index()      

    logger2.info(gdp_yoy[:10:2])
    logger2.info(gdp_qoq[:10:2])

    # Graph: Shanghai vs GDP (Gross domestic product)
    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(gdp_yoy['Date'], (gdp_yoy['Actual/YoY']), label='GDP/YoY', linewidth=1, \
                    linestyle='--')
    lns2 = ax1.plot(gdp_qoq['Date'], (gdp_qoq['Actual/QoQ']), label='GDP/QoQ', linewidth=1, \
                    linestyle='--')
    # lns3 = ax1.plot(gdp.index, gdp['Actual/YTD'], color='maroon', label='GDP/YTD', linewidth=2, \
    #                 linestyle='--', marker='x', markersize=4)
    ax2 = ax1.twinx()
    lns4 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)

    plt.title(f"Shanghai vs GDP(YoY, QoQ)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1+lns2
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns4, lns4, loc=1)
    plt.savefig(reports_dir + '/cn_e0210.png')

    return gdp_yoy, gdp_qoq


'''
**2.2 Shanghai vs Industrial Production(YoY, YTD)
주가가 GDP 발표나기 1개월이전정도면 확실히 알고 미리 움직이고 있음.
주가가 미래의 경제를 선반영  
'''
def shanghai_vs_ip(cals, gdp_yoy, gdp_qoq):
    ip_yoy = cals.loc[cals['event'].str.contains('Industrial Production YoY', regex=False)]
    ip_yoy['Date'] = pd.to_datetime(ip_yoy.date).dt.date
    ip_yoy['Actual/YoY'] = ip_yoy['actual'].astype('float')
    ip_yoy = ip_yoy.dropna(subset=['Actual/YoY'])
    ip_yoy['Date'].reset_index()      
    logger2.info(ip_yoy[:10:2])

    # Graph: Shanghai vs Industrial Production
    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(gdp_yoy['Date'], gdp_yoy['Actual/YoY'], label='GDP/YoY', linewidth=1, \
                    linestyle='--')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)

    plt.title(f"Shanghai vs Industrial Production(YoY, YTD)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0220.png')


'''
2.3 Shanghai vs House Prices (YoY)
'''
def shanghai_vs_house(cals):
    # House Prices (YoY)
    house = cals.loc[cals['event'].str.contains('House Price Index YoY', regex=False)]
    house['Date'] = pd.to_datetime(house.date).dt.date
    house['Actual/YTD'] = house['actual'].astype('float')
    house = house.dropna(subset=['Actual/YTD'])
    house['Date'].reset_index()     
    logger2.info(house[:10:2])

    # Graph: Shanghai vs House Prices (YoY)
    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(house['Date'], house['Actual/YTD'], color='maroon', label='House Prices (YoY)', linewidth=2, \
                    linestyle='--', marker='x', markersize=4)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)

    plt.title(f"Shanghai vs House Prices (YoY)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis

    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0230.png')


'''
3. 환율/통관수출입/외환
3.1 Shanghai vs Export/Import/Balance
'''
def shanghai_vs_eximport(cals):
    # Export
    china_epi = cals.loc[cals['event'].str.contains('Exports YoY',regex=False)]
    china_epi['Date'] = pd.to_datetime(china_epi.date).dt.date
    china_epi['Actual/YoY'] = china_epi['actual'].astype('float')
    china_epi = china_epi.dropna(subset=['Actual/YoY'])
    china_epi['Date'].reset_index()     
    # Import
    china_ipi = cals.loc[cals['event'].str.contains('Imports YoY',regex=False)]
    china_ipi['Date'] = pd.to_datetime(china_ipi.date).dt.date
    china_ipi['Actual/YoY'] = china_ipi['actual'].astype('float')
    china_ipi = china_ipi.dropna(subset=['Actual/YoY'])
    china_ipi['Date'].reset_index()     
    # Trade Balance
    china_tb = cals.loc[cals['event'].str.contains('Balance of Trade',regex=False)]
    china_tb['Date'] = pd.to_datetime(china_tb.date).dt.date
    china_tb['Actual/YoY'] = china_tb['actual'].astype('float')
    china_tb = china_tb.dropna(subset=['Actual/YoY'])
    china_tb['Date'].reset_index()     
    logger2.info(china_tb[china_tb['actual'] < 20])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(china_epi['Date'], (china_epi['Actual/YoY']), color='gold', label='Exports', linewidth=1, \
                    linestyle='--')
    lns2 = ax1.plot(china_ipi['Date'], (china_ipi['Actual/YoY']), color='orange', label='Imports', linewidth=1, \
                    linestyle='--')
    lns3 = ax1.plot(china_tb['Date'], (china_tb['Actual/YoY']), color='maroon', label='Trade Balance', linewidth=1, \
                    linestyle='--', marker='x', markersize=4)
    ax2 = ax1.twinx()
    lns4 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs Trade Balance", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1+lns2+lns3
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns4, lns4, loc=1)
    plt.savefig(reports_dir + '/cn_e0310.png')

'''
3.2 Shanghai vs Dollar Reserve
'''
def shanghai_vs_dollar(cals):
    fx_res = cals.loc[cals['event'].str.contains('Foreign Exchange Reserves',regex=False)]
    fx_res['Date'] = pd.to_datetime(fx_res.date).dt.date
    fx_res['Actual(Bil)'] = fx_res['actual'].astype('float')
    fx_res['Change(Bil)'] = fx_res['Actual(Bil)'] - fx_res['Actual(Bil)'].shift(1)
    fx_res = fx_res.dropna(subset=['Actual(Bil)'])
    fx_res = fx_res.dropna(subset=['Change(Bil)'])    
    fx_res['Date'].reset_index()    
    logger2.info(fx_res[-5:])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(fx_res['Date'], fx_res['Actual(Bil)'], label='Dollar Reserve', linewidth=2, color='maroon',\
            linestyle='-', marker='o', markersize=4)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs Dollar Reserve", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    ax1.grid()
    ax1.legend(lns1, lns1, loc=2)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0320.png')


'''
4. 물가
4.1 shanghai vs CPI + PPI
생산자물자 PPI 가 오르고, 소비자물자 CPI 오르면 얼마후 주가가 상승했다가 큰 하락함 ???? 맞음 ???
'''
def shanghai_vs_cpi_ppi(cals):
    # PPI: The Producer Price Index (PPI)
    ppi_yoy = cals.loc[cals['event'].str.contains('PPI YoY', regex=False)]
    ppi_yoy['Date'] = pd.to_datetime(ppi_yoy.date).dt.date
    ppi_yoy['Actual'] = ppi_yoy['actual'].astype('float')
    ppi_yoy = ppi_yoy.dropna(subset=['Actual'])
    ppi_yoy['Date'].reset_index()     
    logger2.info(ppi_yoy[:5])

    # The Consumer Price Index (CPI) 
    cpi_yoy = cals.loc[cals['event'].str.contains('Inflation Rate YoY', regex=False)]
    cpi_mom = cals.loc[cals['event'].str.contains('Inflation Rate MoM', regex=False)]
    cpi_yoy['Date'] = pd.to_datetime(cpi_yoy.date).dt.date
    cpi_mom['Date'] = pd.to_datetime(cpi_mom.date).dt.date
    cpi_yoy['Actual/YoY'] = cpi_yoy['actual'].astype('float')
    cpi_mom['Actual/MoM'] = cpi_mom['actual'].astype('float')
    cpi_yoy = cpi_yoy.dropna(subset=['Actual/YoY'])
    cpi_yoy['Date'].reset_index()
    cpi_mom = cpi_mom.dropna(subset=['Actual/MoM'])
    cpi_mom['Date'].reset_index()         
    logger2.info(cpi_yoy[:5])
    logger2.info(cpi_mom[:5])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(cpi_yoy['Date'], cpi_yoy['Actual/YoY'], color='blue', label='CPI (YoY)', linewidth=1, \
                    linestyle='--')
    lns2 = ax1.plot(cpi_mom['Date'], cpi_mom['Actual/MoM'], color='orange', label='CPI (MoM)', linewidth=1, \
                    linestyle='--')
    lns3 = ax1.plot(ppi_yoy['Date'], ppi_yoy['Actual'], color='maroon', label='PPI (YoY)', linewidth=1, \
                    linestyle='--', marker='x', markersize=4)
    ax2 = ax1.twinx()
    lns4 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs (CPI + PPI)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1+lns2+lns3
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=2)
    ax2.legend(lns4, lns4, loc=1)
    plt.savefig(reports_dir + '/cn_e0410.png')


'''
5. 투자
5.1 PMI (Purchasing Managers' Index)
PMI 50보다 크면 확장, 50보다 작으면 침체
'''
def pmi(cals):
    # Chinese Composite PMI
    # pmi_comp = cals.loc[cals['event'].str.contains('Chinese Composite PMI', regex=False)]
    # pmi_comp['Date'] = pd.to_datetime(pmi_comp.index, dayfirst=True)
    # pmi_comp['Actual'] = pmi_comp['actual'].astype('float')
    # display(pmi_comp[-5:])

    # Manufacturing PMI
    pmi_manuf = cals.loc[cals['event'].str.contains('NBS Manufacturing PMI', regex=False)]
    pmi_manuf['Date'] = pd.to_datetime(pmi_manuf.date).dt.date
    pmi_manuf['Actual'] = pmi_manuf['actual'].astype('float')
    pmi_manuf = pmi_manuf.dropna(subset=['Actual'])
    pmi_manuf['Date'].reset_index()     
    logger2.info(pmi_manuf[:5])

    # Non-Manufacturing PMI
    pmi_non_manuf = cals.loc[cals['event'].str.contains('NBS Non Manufacturing PMI', regex=False)]
    pmi_non_manuf['Date'] = pd.to_datetime(pmi_non_manuf.date).dt.date
    pmi_non_manuf['Actual'] = pmi_non_manuf['actual'].astype('float')
    pmi_non_manuf = pmi_non_manuf.dropna(subset=['Actual'])
    pmi_non_manuf['Date'].reset_index()       
    logger2.info(pmi_non_manuf[:5])

    # Caixin Manufacturing PMI
    pmi_manuf_caixin = cals.loc[cals['event'].str.contains('Caixin Manufacturing PMI', regex=False)]
    pmi_manuf_caixin['Date'] = pd.to_datetime(pmi_manuf_caixin.date).dt.date
    pmi_manuf_caixin['Actual'] = pmi_manuf_caixin['actual'].astype('float')
    pmi_manuf_caixin = pmi_manuf_caixin.dropna(subset=['Actual'])
    pmi_manuf_caixin['Date'].reset_index()         
    logger2.info(pmi_manuf_caixin[:5])

    # Caixin Services PMI
    pmi_service_caixin = cals.loc[cals['event'].str.contains('Caixin Services PMI', regex=False)]
    pmi_service_caixin['Date'] = pd.to_datetime(pmi_service_caixin.date).dt.date
    pmi_service_caixin['Actual'] = pmi_service_caixin['actual'].astype('float')
    pmi_service_caixin = pmi_service_caixin.dropna(subset=['Actual'])
    pmi_service_caixin['Date'].reset_index()      
    logger2.info(pmi_service_caixin[-5:])


    fig, ax1 = plt.subplots(figsize=(16,8))
    # lns1 = ax1.plot(pmi_comp['Date'], (pmi_comp['Actual']), color='maroon', label='Chinese Composite PMI', linewidth=2, \
    #                 linestyle='-', marker='x', markersize=4)
    lns2 = ax1.plot(pmi_manuf['Date'], pmi_manuf['Actual'], label='Manufacturing PMI', linewidth=1, \
                    linestyle='--')
    lns3 = ax1.plot(pmi_non_manuf['Date'], pmi_non_manuf['Actual'], label='Non-Manufacturing PMI', linewidth=1, \
                    linestyle='--')
    lns4 = ax1.plot(pmi_manuf_caixin['Date'], pmi_manuf_caixin['Actual'], label='Caixin Manufacturing PMI', linewidth=1, \
                    linestyle='-.')
    # lns5 = ax1.plot(pmi_service_caixin['Date'], pmi_service_caixin['Actual'], label='Caixin Services PMI', linewidth=1, \
    #                 linestyle='-.')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs PMIs", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    ax1.axhline(y=50, linestyle='--', color='red', linewidth=2)
    lns = lns2+lns3+lns4
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns6, lns6, loc=1)
    plt.savefig(reports_dir + '/cn_e0510.png')


'''
5.2 Chinese Industrial profit (YoY, YTD)
'''
def indu_profit(cals):
    # Chinese Industrial profit  (YoY) 
    indu_profit_yoy = cals.loc[cals['event'].str.contains('Industrial Profits YoY', regex=False)]
    indu_profit_yoy['Date'] = pd.to_datetime(indu_profit_yoy.date).dt.date
    indu_profit_yoy['Actual'] = indu_profit_yoy['actual'].astype('float')
    indu_profit_yoy = indu_profit_yoy.dropna(subset=['Actual'])
    indu_profit_yoy['Date'].reset_index()     
    logger2.info(indu_profit_yoy[:5])

    # Chinese Industrial profit YTD
    indu_profit_ytd = cals.loc[cals['event'].str.contains('Industrial Profits (YTD) YoY', regex=False)]
    indu_profit_ytd['Date'] = pd.to_datetime(indu_profit_ytd.date).dt.date
    indu_profit_ytd['Actual'] = indu_profit_ytd['actual'].astype('float')
    indu_profit_ytd = indu_profit_ytd.dropna(subset=['Actual'])
    indu_profit_ytd['Date'].reset_index()       
    logger2.info(indu_profit_ytd[:5])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(indu_profit_yoy['Date'], indu_profit_yoy['Actual'], color='maroon', label='Chinese Industrial profit  (YoY)', linewidth=2, \
                    linestyle='-', marker='x', markersize=4)
    lns2 = ax1.plot(indu_profit_ytd['Date'], indu_profit_ytd['Actual'], label='Chinese Industrial profit YTD', linewidth=1, \
                    linestyle='-')
    ax2 = ax1.twinx()
    lns6 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs PMIs", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1+lns2
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns1, lnses, loc=3)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0520.png')


'''
5.3 Foreign direct investment
'''
def foreign_invest(cals):
    # fdi (Foreign direct investment)
    fdi = cals.loc[cals['event'].str.contains('FDI (YTD) YoY', regex=False)]
    fdi['Date'] = pd.to_datetime(fdi.date).dt.date
    fdi['Actual'] = fdi['actual'].astype('float')
    fdi = fdi.dropna(subset=['Actual'])
    fdi['Date'].reset_index()     
    logger2.info(fdi[:5])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(fdi['Date'], (fdi['Actual']), color='maroon', label='FDI (Foreign direct investment)', linewidth=2, \
                    linestyle='-', marker='x', markersize=4)
    ax2 = ax1.twinx()
    lns6 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs FDI(Foreign direct investment)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    plt.savefig(reports_dir + '/cn_e0530.png')


'''
5.4 Fixed Asset Investment (YoY)
'''
def fixed_asset_invest(cals):
    # Fixed Asset Investment (YoY) 
    fai = cals.loc[cals['event'].str.contains('Fixed Asset Investment (YTD) YoY', regex=False)]
    fai['Date'] = pd.to_datetime(fai.date).dt.date
    fai['Actual'] = fai['actual'].astype('float')
    fai = fai.dropna(subset=['Actual'])
    fai['Date'].reset_index()     
    logger2.info(fai[:5])

    fig, ax1 = plt.subplots(figsize=(16,4))
    lns1 = ax1.plot(fai['Date'], fai['Actual'], color='maroon', label='Fixed Asset Investment (YoY) ', linewidth=2, \
                    linestyle='-', marker='x', markersize=4)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(shanghai_shares.index, shanghai_shares['Close'], color='green', label='Shanghai', linewidth=1)
    plt.title(f"Shanghai vs Fixed Asset Investment (YoY) ", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    lns = lns1
    lnses = [l.get_label() for l in lns]
    ax1.grid()
    ax1.legend(lns, lnses, loc=3)
    ax2.legend(lns2, lns2, loc=1)
    plt.savefig(reports_dir + '/cn_e0540.png')




'''
Main Fuction
'''

if __name__ == "__main__":
    cals = eco_calendars(from_date, to_date_2)  # calendars
    shanghai_shares, szse_shares = shanghai_szse_vs_yuan(from_date_MT, to_date)
    shanghai_vs_loan(cals)
    shanghai_vs_m2(cals)
    house_loan(cals)
    yuan_exchange_rate(from_date_MT)
    gdp_yoy, gdp_qoq = shanghai_vs_gpd(cals)
    shanghai_vs_ip(cals, gdp_yoy, gdp_qoq)
    shanghai_vs_house(cals)
    shanghai_vs_eximport(cals)
    shanghai_vs_dollar(cals)
    shanghai_vs_cpi_ppi(cals)   
    pmi(cals)
    indu_profit(cals)
    foreign_invest(cals)
    fixed_asset_invest(cals)

