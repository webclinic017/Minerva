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
1. Leading Indicators OECD
1.1 Leading indicators: CLI (Composite leading indicator)
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
    plt.savefig(reports_dir + '/global_e0200.png')


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
    plt.savefig(reports_dir + '/global_e0300.png')


'''
1.4 Sovereign CDS
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
Main Fuction
'''

if __name__ == "__main__":

    # 1. Leading Indicators OECD
    cli()
    m1()
    cpi()
    cds()