'''
Prgram 명: Glance of Global countries
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
글로벌 경쟁국들의 투자를 위한 경제지표(거시/미시) 부문 엿보기
* investing.com/calendar 포함
History
20231111  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''

import requests
from bs4 import BeautifulSoup as bs

# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
to_date_2 = pd.to_datetime(to_date2)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()


'''
1. Economics Area
1.1 Leading Indicators OECD: CLI (Composite leading indicator)
'''

def cli():
    CLI_OECD_Total = fred.get_series(series_id='OECDLOLITONOSTSAM', observation_start=from_date_MT)
    CLI_OECD_Total_Plus_Six = fred.get_series(series_id='ONMLOLITONOSTSAM', observation_start=from_date_MT)
    CLI_Usa = fred.get_series(series_id='USALOLITONOSTSAM', observation_start=from_date_MT)
    CLI_Korea = fred.get_series(series_id='KORLOLITONOSTSAM', observation_start=from_date_MT)
    CLI_China = fred.get_series(series_id='CHNLOLITONOSTSAM', observation_start=from_date_MT)
    CLI_Euro = fred.get_series(series_id='EA19LOLITONOSTSAM', observation_start=from_date_MT)

    logger2.info('##### Leading indicators: CLI (Composite leading indicator) #####')
    logger2.info('CLI_OECD_Total: \n' + str(CLI_OECD_Total[-3:]))
    logger2.info('CLI_OECD_Total_Plus_Six: \n' + str(CLI_OECD_Total_Plus_Six[-3:]))
    logger2.info('CLI_Usa: \n' + str(CLI_Usa[-3:]))
    logger2.info('CLI_Korea: \n' + str(CLI_Korea[-3:]))
    logger2.info('CLI_China: \n' + str(CLI_China[-3:]))
    logger2.info('CLI_Euro: \n' + str(CLI_Euro[-3:]))  

    plt.figure(figsize=(15,6))
    plt.title(f"Leading indicators: CLI (Composite leading indicator)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(CLI_OECD_Total, label='CLI_OECD_Total')
    plt.plot(CLI_OECD_Total_Plus_Six, label='CLI_OECD_Total_Plus_Six')
    plt.plot(CLI_Usa, label='CLI_Usa')
    plt.plot(CLI_Korea, label='CLI_Korea')
    plt.plot(CLI_China, label='CLI_China')
    plt.plot(CLI_Euro, label='CLI_Euro')
    plt.grid()
    plt.legend()
    plt.savefig(reports_dir + '/global_e0100.png')


'''
1.2 M1
'''
def m1():

    fig, ax = plt.subplots(figsize=(18, 6 * 2))
    # logn term view
    usa = fred.get_series(series_id='WM1NS', observation_start=from_date_LT)
    china = fred.get_series(series_id='MYAGM2CNM189N', observation_start=from_date_LT)
    japan = fred.get_series(series_id='MANMM101JPM189S', observation_start=from_date_LT)
    euro = fred.get_series(series_id='MANMM101EZM189S', observation_start=from_date_LT)
    korea = fred.get_series(series_id='MANMM101KRM189S', observation_start=from_date_LT)

    plt.subplot(2, 1, 1)
    plt.title(f"M1 normalized over the countries over long term", fontdict={'fontsize':20, 'color':'g'})
    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.axhline(y=2, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.plot(normalize(usa), label='usa')
    plt.plot(normalize(china), label='china')
    plt.plot(normalize(japan), label='japan')
    plt.plot(normalize(euro), label='euro')
    plt.plot(normalize(korea), label='korea')
    plt.grid()
    plt.legend()

    # mid term view
    usa = fred.get_series(series_id='WM1NS', observation_start=from_date_MT)
    china = fred.get_series(series_id='MYAGM2CNM189N', observation_start=from_date_MT)
    japan = fred.get_series(series_id='MANMM101JPM189S', observation_start=from_date_MT)
    euro = fred.get_series(series_id='MANMM101EZM189S', observation_start=from_date_MT)
    korea = fred.get_series(series_id='MANMM101KRM189S', observation_start=from_date_MT)
    plt.subplot(2, 1, 2)
    plt.title(f"M1 normalized over the countries over mid term", fontdict={'fontsize':20, 'color':'g'})
    # Covid-19 Crisis
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)

    plt.axhline(y=2, linestyle='--', color='red', linewidth=1, label='2% Target Rate')
    plt.plot(normalize(usa), label='usa')
    plt.plot(normalize(china), label='china')
    plt.plot(normalize(japan), label='japan')
    plt.plot(normalize(euro), label='euro')
    plt.plot(normalize(korea), label='korea')
    plt.grid()
    plt.legend()

    # 이미지 파일로 저장
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/global_e0120.png')


'''
1.3 CPI (Consumer Price Indices)
'''
def cpi():
    cpi_us = fred.get_series(series_id='CPIAUCSL', observation_start=from_date_MT).pct_change(periods=12)*100
    cpi_japan = fred.get_series(series_id='JPNCPIALLMINMEI', observation_start=from_date_MT).pct_change(periods=12)*100
    cpi_euro = fred.get_series(series_id='CP0000EZ19M086NEST', observation_start=from_date_MT).pct_change(periods=12)*100
    cpi_korea = fred.get_series(series_id='KORCPIALLMINMEI', observation_start=from_date_MT).pct_change(periods=12)*100
    cpi_china = fred.get_series(series_id='CHNCPIALLMINMEI', observation_start=from_date_MT).pct_change(periods=12)*100

    logger2.info('##### CPI (Consumer Price Indices) #####')
    logger2.info('cpi_us: \n' + str(cpi_us[-3:]))
    logger2.info('cpi_japan: \n' + str(cpi_japan[-3:]))
    logger2.info('cpi_euro: \n' + str(cpi_euro[-3:]))  
    logger2.info('cpi_korea: \n' + str(cpi_korea[-3:]))
    logger2.info('cpi_china: \n' + str(cpi_china[-3:]))

    plt.figure(figsize=(18,4))
    plt.title(f"Consumer Price Indices for Countries", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)# Covid-19 Crisis
    plt.plot(cpi_us, label='CPI for U.S. City')
    plt.plot(cpi_japan, label='CPI for Japan')
    plt.plot(cpi_euro, label='CPI for Euro 19 countries')
    plt.plot(cpi_korea, label='CPI for South Korea')
    plt.plot(cpi_china, label='CPI for China')
    plt.legend()
    plt.savefig(reports_dir + '/global_e0130.png')





'''
2. Market Area
2.1 Sovereign CDS
'''
def cds():
    page = requests.get("https://www.worldgovernmentbonds.com/sovereign-cds/")
    soup = bs(page.text, "html.parser")

    # 제거하려는 태그 선택
    tag_to_remove = soup.find('tfoot')
    # 태그 제거
    tag_to_remove.decompose()

    tables = soup.find_all('table')
    # 멀티 헤더의 첫번째 헤더 제거
    cdses = pd.read_html(str(tables))[0]
    column_to_remove = cdses.columns[0]
    cdses.drop(columns=column_to_remove, inplace=True)
    cdses.columns = cdses.columns.droplevel(0)

    logger2.info('##### Sovereign CDS (Credit Default Swap) #####')
    logger2.info('cdses: \n' + str(cdses))



'''
3. Business Area
3.1 Containerized Freight Index
- 해상운임지수: 경기가 다시 활성화 되는지 여부 모니터링 (20220906)
'''
def container_Freight():
    # CCFI (China Containerized Freight Index)
    # 중국컨테이너운임지수는 중국 교통부가 주관하고 상하이 항운교역소가 집계하는 중국발 컨테이너운임지수로 1998년 4월 13일 처음 공시되었다. 
    # 세계컨테이너시황을 객관적으로 반영한 지수이자 중국 해운시황을 나타내는 주요 지수로 평가받고 있다.
    # 1998년 1월 1일을 1,000으로 산정하며 중국의 항구를 기준으로 11개의 주요 루트별 운임을 산정하며, 16개 선사의 운임정보를 기준으로 
    # 매주 금요일에 발표를 하고 있다.
    page = requests.get("https://www.kcla.kr/web/inc/html/4-1_2.asp")
    soup = bs(page.text, "html.parser")
    table = soup.find_all('table')
    df_ccfi = pd.read_html(str(table))[0]

    df_ccfi = df_ccfi.T
    df_ccfi.drop([0], inplace=True)
    df_ccfi[1] = df_ccfi[1].astype('float')
    df_ccfi.rename(columns={0:'Date', 1:'Idx'}, inplace=True)
    df_ccfi['Date']= pd.to_datetime(df_ccfi['Date'])
    df_ccfi.set_index('Date', inplace=True)

    # SCFI (Shanghai Containerized Freight Index)
    # 상하이컨테이너 운임지수는 상하이거래소(Shanghai Shipping Exchange: SSE)에서 2005년 12월 7일부터 상하이 수출컨테이너 운송시장의 
    # 15개 항로의 스팟(spot) 운임을 반영한 운임지수이다. 기존에는 정기용선운임을 기준으로 하였으나 2009년 10월 16일부터는 20ft 컨테이너(TEU)당 
    # 미달러(USD)의 컨테이너 해상화물운임에 기초하여 산정하고 있다.
    # 운송조건은 CY-CY조건이며 컨테이너의 타입과 화물의 상세는 General Dry Cargo Container로 한정짓고 있고, 개별항로의 운임율은 각 항로의 
    # 모든 운임율의 산술평균이며 해상운송에 기인한 할증 수수료가 포함되어 있다. 운임정보는 정기선 선사와 포워더를 포함한 CCFI의 패널리스트들에게 
    # 제공받고 있다.
    page = requests.get("https://www.kcla.kr/web/inc/html/4-1_3.asp")
    soup = bs(page.text, "html.parser")
    table = soup.find_all('table')
    df_scfi = pd.read_html(str(table))[0]

    df_scfi = df_scfi.T
    df_scfi.drop([0], inplace=True)
    df_scfi[1] = df_scfi[1].astype('float')
    df_scfi.rename(columns={0:'Date', 1:'Idx'}, inplace=True)
    df_scfi['Date']= pd.to_datetime(df_scfi['Date'])
    df_scfi.set_index('Date', inplace=True)

    # HRCI (Howe Robinson Container Index)
    # 영국의 대표적인 해운컨설팅 및 브로커社인 Howe Robinson社가 발표하는 컨테이너 지수로서 선박을 하루 용선하는 데 소요되는 비용에 대한
    # 컨테이너 시장 용선요율을 나타내고 있다. 이 회사는 1883년 설립되었으며 컨테이너선과 벌크선에 대한 세계에서 가장 크고 독립적인 중개회사 중 
    # 하나로 1997년 1월 1일을 1,000으로 놓고 매주 발표하고 있다.
    page = requests.get("https://www.kcla.kr/web/inc/html/4-1_4.asp")
    soup = bs(page.text, "html.parser")
    table = soup.find_all('table')
    df_hrci = pd.read_html(str(table))[0]

    df_hrci = df_hrci.T
    df_hrci.drop([0], inplace=True)
    df_hrci[1] = df_hrci[1].astype('float')
    df_hrci.rename(columns={0:'Date', 1:'Idx'}, inplace=True)
    df_hrci['Date']= pd.to_datetime(df_hrci['Date'])
    df_hrci.set_index('Date', inplace=True)


    # BDI (Baltic Dry Index)
    # 발틱운임지수는 발틱해운거래소에서 1999년 11월 1일부터 사용되었으며 1985년부터 건화물(dry cargo)의 운임지수로 사용이 되어온 
    # BFI(Baltic Freight Index)를 대체한 종합운임지수로 1985년 1월 4일을 1,000으로 산정하여 선박의 형태에 따라 발표하고 있다.
    # 선형에 따라 Baltic Capesize Index(BCI), Baltic Panamax Index(BPI), Baltic Supramax Index(BSI), 
    # Baltic Handysize Index(BHSI) 등으로 구성되어 있으며, BDI는 이러한 선형별 정기용선의 4가지 지수를 동일한 가중으로 평균을 산출한 다음 
    # BDI factor를 곱하여 산출하고 있다.
    page = requests.get("https://www.kcla.kr/web/inc/html/4-1_5.asp")
    soup = bs(page.text, "html.parser")
    table = soup.find_all('table')
    df_bdi = pd.read_html(str(table))[0]

    df_bdi = df_bdi.T
    df_bdi.drop([0], inplace=True)
    df_bdi[1] = df_bdi[1].astype('float')
    df_bdi.rename(columns={0:'Date', 1:'Idx'}, inplace=True)
    df_bdi['Date']= pd.to_datetime(df_bdi['Date'])
    df_bdi.set_index('Date', inplace=True)

    plt.figure(figsize=(15,5))
    plt.title(f"Containerized Freight Index", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_ccfi, label='China') 
    plt.plot(df_scfi, label='Shanghai') 
    plt.plot(df_hrci, label='Howe Robinson') 
    plt.plot(df_bdi, label='Baltic Dry') 
    plt.legend()
    plt.savefig(reports_dir + '/global_e0310.png')



'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    1. Economic Area
    '''
    cli()
    m1()
    cpi()

    '''
    2. Market Area
    '''
    cds()

    '''
    3. Business Area
    '''
    container_Freight()