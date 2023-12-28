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
import yfinance as yf
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
    CLI_Germany = fred.get_series(series_id='DEULOLITOTRSTSAM', observation_start=from_date_MT)

    logger2.info('##### Leading indicators: CLI (Composite leading indicator) #####')
    logger2.info('CLI_OECD_Total: \n' + str(CLI_OECD_Total[-3:]))
    logger2.info('CLI_OECD_Total_Plus_Six: \n' + str(CLI_OECD_Total_Plus_Six[-3:]))
    logger2.info('CLI_Usa: \n' + str(CLI_Usa[-3:]))
    logger2.info('CLI_Korea: \n' + str(CLI_Korea[-3:]))
    logger2.info('CLI_China: \n' + str(CLI_China[-3:]))
    logger2.info('CLI_Germany: \n' + str(CLI_Germany[-3:]))  

    plt.figure(figsize=(15,6))
    plt.title(f"Leading indicators: CLI (Composite leading indicator)", fontdict={'fontsize':20, 'color':'g'})
    plt.axvspan(datetime(2020,3,3), datetime(2020,3,30), facecolor='gray', edgecolor='gray', alpha=0.3)
    plt.plot(CLI_OECD_Total, label='CLI_OECD_Total')
    plt.plot(CLI_OECD_Total_Plus_Six, label='CLI_OECD_Total_Plus_Six')
    plt.plot(CLI_Usa, label='CLI_Usa')
    plt.plot(CLI_Korea, label='CLI_Korea')
    plt.plot(CLI_China, label='CLI_China')
    plt.plot(CLI_Germany, label='CLI_Germany')
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
4. calculate trend
- country growth, market growth, business growth
'''
class CalTrend():
    '''
    1) cal_trend
      - cal_country_growth
      - cal_market_growth
      - cal_busi_growth
    2) cal_country_growth
      - get_GDP_rate
      - get_inflation
      - get_export
    3) cal_market_growth
      - 
    4) cal_busi_growth
      - 
    '''

    def __init__(self):
        database = 'Economics.db'
        db_file = 'database/' + database
        conn, engine = create_connection(db_file)
        self.conn = conn
        self.engine = engine


    def get_GDP_rate(self, conn, country_sign): # rate 로 측정되는 값은 'actual' 값을 기준으로 가감을 하고,             
            if country_sign in ['US', 'JP']:
                gdp = pd.read_sql_query(f"SELECT * FROM Calendars WHERE event like 'GDP Growth Rate QoQ%'  AND country = '{country_sign}' \
                    ORDER BY date DESC LIMIT 2", conn)
                print(gdp)
            elif country_sign in ['KR', 'CN', 'DE', 'IN', 'SG', 'BR', 'IN']:
                gdp = pd.read_sql_query(f"SELECT * FROM Calendars WHERE event like '%GDP Growth Rate YoY%'  AND country = '{country_sign}' \
                                        ORDER BY date DESC LIMIT 2", conn)
                print(gdp)
            else:
                print(f'Country Sign is not found: {country_sign}' )

            
            try:
                if np.isnan(gdp['estimate'][0]) or np.isnan(gdp['actual'][0]):
                    idx = 1
                else:
                    idx = 0
            except:
                result = gdp['actual'][0]
                return result
            
            # 보정 1: 지난 값대비 변화율로 가감점
            mul_1 = (gdp['change'][idx] / 100)
            
            # 보정 2: 예측치 대비 실측치로 가감점
            mul_2 = (gdp['actual'][idx] - gdp['estimate'][idx]) / 100
            
            logger2.debug('mul_1: ' + str(mul_1))
            logger2.debug('mul_2: ' + str(mul_2))
            
            result = gdp['actual'][idx] + mul_1 + mul_2
            
            return result
    
    
    def get_inflation(self, conn, country_sign): # inflation mom : price 로 측정되는 값은 'change' 값을 기준으로 가감함.
        inflation = pd.read_sql_query(f"SELECT * FROM Calendars WHERE event like 'Inflation Rate YoY%' \
                                      AND country = '{country_sign}'\
                                      ORDER BY date DESC LIMIT 2", conn)

        print(inflation)

        try:
            if np.isnan(inflation['estimate'][0]) or np.isnan(inflation['actual'][0]):
                idx = 1
            else:
                idx = 0
        except:
            result = inflation['actual'][0]
            return result
        
        # 보정 1: 지난 월단위값 대비 변화율
        mul_1 = (inflation['change'][idx] / 100)
        
        # 보정 2: 예측치 대비 실측치로 +- 20% 보정
        mul_2 = (inflation['actual'][idx] - inflation['estimate'][idx]) / 100
    
        logger2.debug('mul_1: ' + str(mul_1))
        logger2.debug('mul_2: ' + str(mul_2))    

        result = inflation['actual'][idx] + mul_1 + mul_2
        
        return result
    
    
    def get_export(self, conn, country_sign): # export yoy  : price 로 측정되는 값은 'change' 값을 기준으로 가감함.
        if country_sign in ['US']:            
            export = pd.read_sql_query(f"SELECT * FROM Calendars WHERE country = '{country_sign}' \
                AND event like 'Export Prices YoY%'  ORDER BY date DESC LIMIT 2", conn)
        elif country_sign in ['KR', 'JP', 'CN', 'SG',]:   # SG ..... ??? 넘어가네... 
            export = pd.read_sql_query(f"SELECT * FROM Calendars WHERE country = '{country_sign}' \
                AND event like 'Exports YoY%'  ORDER BY date DESC LIMIT 2", conn)
        elif country_sign in ['DE',]:    
            export = pd.read_sql_query(f"SELECT * FROM Calendars WHERE country = '{country_sign}' \
                AND event like 'Exports MoM%'  ORDER BY date DESC LIMIT 2", conn)  
        elif country_sign in ['IN',]:    
            export = pd.read_sql_query(f"SELECT * FROM Calendars WHERE country = '{country_sign}' \
                AND event like 'Exports%'  ORDER BY date DESC LIMIT 2", conn)              
        else:
            logger2.info(f'Country Sign is not found: {country_sign}')

        print(export)
        
        try:
            if np.isnan(export['estimate'][0]) or np.isnan(export['actual'][0]):
                idx = 1
            else:
                idx = 0
                
            # 보정 1: 지난 월단위값 대비 변화율
            mul_1 = (export['change'][idx] / 100)

            # 보정 2: 예측치 대비 실측치로 보정
            mul_2 = (export['actual'][idx] - export['estimate'][idx]) / 100                
        except:
            mul_1 = 0
            mul_2 = 0
            idx = 0
            pass
        
        logger2.debug('mul_1: ' + str(mul_1))
        logger2.debug('mul_2: ' + str(mul_2))           
        
        result = export['actual'][idx] + mul_1 + mul_2
        
        return result

    ########################################################################### 
    '''
    국가단위의 현재와 미래의 성장율을 도출하는 루틴으로,
    1. 현재: realGDP실적 성장율(+- 예측치대비)
    1.1 fmp.calendar.com 에서 추출한 realGDP (= nominalGDP - Inflation) 로 계산한 값
        - 잠재성장률대비 real GDP YoY, (include KR.Export) = nominal GDP - Inflation
    1.2 macrovar.com 에서 추출한 realGDP

    2. 미래: (LEI 성장율 예측 (각종 기관들의 값 평균치 활용, IMF, OECD...) + ML perdict) / 2
    2.1 IMF 전망치: 
    '''        
    def cal_country_growth(self, conn, country_sign:str, month_term:int=0):
       
        gdp = self.get_GDP_rate(conn, country_sign)
        inflation = self.get_inflation(conn, country_sign)
        if country_sign in ['EU', 'SG']:
            export = 1
        else:
            export = self.get_export(conn, country_sign)

        logger2.info(' ')        
        logger2.info(f'##### gdp: {str(gdp)} %')           
        logger2.info(f'##### inflation: {str(inflation)} %')
        logger2.info(f'##### export: {str(export)} %')            
        
        realGDP_rate = gdp - inflation*0.3 
        if country_sign in ['KR', 'JP', 'CN']: # 수출 주도형 국가: 한국, 일본, 중국
            result = gdp - inflation*0.3 + export*0.3 # GDP 대비 inflation 비중이 30% 가정, GDP 대비 무역비중이 30% 가정 
        else:
            result = gdp - inflation*0.3 + export*0.03

        return result
    
    ###########################################################################    
    '''
    business: M2(증가량 배수) 와 자산시장별 변화율 비교
        - 주식, 채권, 원자재, 부동산, 현금
        - US: S&P500/Nasdaq, 국채3년/10년/30년 수익률, GOLD/OIL, LITZ, DOLLOAR
        - KR: KOSPI/KOSDAQ, 국채3년/10년/30년 수익률, 금현물, LITZ, WON
        - JP: NIKKEI, x, GOLD, LITZ, YEN
        - EU: EUROXXX, BONDS, x, x, x
        - CN: SHANHAE/SIMCHUN, x, x, x, x
        - IN: NIFTY, x, x, x, x
    '''
    def cal_market_growth(self, conn, country_sign:str, market_name:str, month_term:int):
    
        try:
            # 2.1 M2
            m2 = pd.read_sql_query(f"SELECT * FROM Indicators WHERE Indicator like '%M2%' AND \
            Country = '{country_sign}' ORDER BY date DESC LIMIT 1", conn)
            m2_growth = m2['YOY'][0]
            if m2['Trend'][0] == 'UP':
                m2_growth = m2_growth * 1.1
            else:
                m2_growth = m2_growth * 0.9
            if m2['Slope'][0] == 'UP':
                m2_growth = m2_growth * 1.05
            else:
                m2_growth = m2_growth * 0.95
        except Exception as e:
            print('Exception: {}'.format(e))  
            
        #2.2 Assets
        if country_sign == 'US':
            if market_name == 'stock':
                ticker = '^SPX'
                ticker = yf.Ticker(ticker)
                sp500 = ticker.history(period='3mo')['Close']                
                sp500 = sp500.dropna()
                growth_1 = (sp500[-1] - sp500[0]) / sp500[0]
                
                ticker = '^IXIC'
                ticker = yf.Ticker(ticker)
                nasdaq = ticker.history(period='3mo')['Close']
                nasdaq = nasdaq.dropna()
                growth_2 = (nasdaq[-1] - nasdaq[0]) / nasdaq[0]

                asset_growth = (growth_1 + growth_2) / 2 * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset')        
            
        elif country_sign == 'KR':
            if market_name == 'stock':
                ticker = '^KS11'
                ticker = yf.Ticker(ticker)
                kospi = ticker.history(period='3mo')['Close']                
                kospi = kospi.dropna()
                growth_1 = (kospi[-1] - kospi[0]) / kospi[0]
                
                ticker = '^KQ11'
                ticker = yf.Ticker(ticker)
                kosdaq = ticker.history(period='3mo')['Close']
                kosdaq = kosdaq.dropna()
                growth_2 = (kosdaq[-1] - kosdaq[0]) / kosdaq[0]

                asset_growth = (growth_1 + growth_2) / 2 * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset')        

        elif country_sign == 'JP':
            if market_name == 'stock':
                ticker = '^N225'
                ticker = yf.Ticker(ticker)
                nikkei = ticker.history(period='3mo')['Close']                
                nikkei = nikkei.dropna()
                growth_1 = (nikkei[-1] - nikkei[0]) / nikkei[0]

                asset_growth = (growth_1) * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset')        

        elif country_sign == 'CN':
            if market_name == 'stock':
                ticker = '000001.SS'
                ticker = yf.Ticker(ticker)
                shanghai = ticker.history(period='3mo')['Close']                
                shanghai = shanghai.dropna()
                growth_1 = (shanghai[-1] - shanghai[0]) / shanghai[0]
                
                ticker = '399001.SZ'
                ticker = yf.Ticker(ticker)
                Shenzhen = ticker.history(period='3mo')['Close']
                Shenzhen = Shenzhen.dropna()
                growth_2 = (Shenzhen[-1] - Shenzhen[0]) / Shenzhen[0]

                asset_growth = (growth_1 + growth_2) / 2 * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset')       

        elif country_sign == 'DE':
            if market_name == 'stock':
                ticker = '^GDAXI'
                ticker = yf.Ticker(ticker)
                dax = ticker.history(period='3mo')['Close']                
                dax = dax.dropna()
                growth_1 = (dax[-1] - dax[0]) / dax[0]

                asset_growth = (growth_1) * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset') 

        elif country_sign == 'EU':
            if market_name == 'stock':
                ticker = '^STOXX50E'
                ticker = yf.Ticker(ticker)
                eu = ticker.history(period='3mo')['Close']                
                eu = eu.dropna()
                growth_1 = (eu[-1] - eu[0]) / eu[0]

                asset_growth = (growth_1) * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset') 

        elif country_sign == 'SG':
            if market_name == 'stock':
                ticker = '^STI'
                ticker = yf.Ticker(ticker)
                ses = ticker.history(period='3mo')['Close']                
                ses = ses.dropna()
                growth_1 = (ses[-1] - ses[0]) / ses[0]

                asset_growth = (growth_1) * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset') 

        elif country_sign == 'BR':
            if market_name == 'stock':
                ticker = '^FTSE'
                ticker = yf.Ticker(ticker)
                ftse = ticker.history(period='3mo')['Close']                
                ftse = ftse.dropna()
                growth_1 = (ftse[-1] - ftse[0]) / ftse[0]

                asset_growth = (growth_1) * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset') 

        elif country_sign == 'IN':
            if market_name == 'stock':
                ticker = '^NSEI'
                ticker = yf.Ticker(ticker)
                nifty50 = ticker.history(period='3mo')['Close']                
                nifty50 = nifty50.dropna()
                growth_1 = (nifty50[-1] - nifty50[0]) / nifty50[0]

                asset_growth = (growth_1) * 100
                
            elif market_name == 'bond':
                y10 = fred.get_series(series_id='DGS10', observation_start=from_date_ST)
                growth = (y10[0] - y10[-1]) / len(y10) 
                print(growth) 
                
            elif market_name == 'commodity':
                gold = fred.get_series(series_id='PPIACO', observation_start=from_date_MT)
                growth = (gold[0] - gold[-1]) / len(gold) 
                print(growth)
            elif market_name == 'currency':
                currency = fred.get_series(series_id='RTWEXBGS', observation_start=from_date_MT)
                growth = (currency[0] - currency[-1]) / len(currency) 
                print(growth)
            else:
                print(f'Error: {country_sign}  {market_name} Not found Asset')                 

        else:
            print('Country sign is not found.')
         
        result = (m2_growth * 0.6) + (asset_growth * 0.4)

        logger2.info(' ')        
        logger2.info(f'##### Market Growth: {result} %')           
        logger2.info(f'##### M2 Growth: {m2_growth} %')
        logger2.info(f'##### Asset Growth: {asset_growth} %')

        return result


    ###########################################################################    
    '''
    business: 각 섹터별 ETF growth 
        - 특별히 업종을 주도하는 섹터만 집중 투자하는 경우만 활용
        - US: spy/qqq, shy/tlt/tmf, gld/bci, o/vnq, dollar
        - kr: kodex200/tiger200it, kodex국고채3년/kosef국고채10년/kbstar kis국고채30년enhanced, 금현물, 맥쿼리인프라, 원
        - jp: 노무라닛케이225etf, yen
        - eu:
        - cn:  
        month_term: 0, 3, 12, 36 개월후 예측치 산정
    '''
    def cal_busi_growth(self, conn, ticker:str, month_term:int):
        '''
            business: 각 섹터별 ETF
            - 특별히 업종을 주도하는 섹터만 집중 투자하는 경우만 활용
            - US: spy/qqq, shy/tlt/tmf, gld/bci, o/vnq, dollar
            - kr: kodex200/tiger200it, kodex국고채3년/kosef국고채10년/kbstar kis국고채30년enhanced, 금현물, 맥쿼리인프라, 원
            - jp: 노무라닛케이225etf, yen
            - eu:
            - cn:  
            month_term: 0, 3, 12, 36 개월후 예측치 산정
        '''
        return 1


    ###########################################################################    
    def cal_trend(self, country:str, market:str, business:str, month_term:int):
        if month_term == 0:
            c_growth = self.cal_country_growth(self.conn, country, month_term)
            m_growth = self.cal_market_growth(self.conn, country, market, month_term)
            b_growth = self.cal_busi_growth(self.conn, business, month_term)
        elif month_term <= 6:  # IMF 와 OECD 평균 전망치 적용
            c_growth = 0
            m_growth = 0
            b_growth = 0
            print('aaa')
        else : # IMF 전망치만 적용
            country_sign2 = 'USA'
            imf_growth = get_country_growth_pjt_byIMF(conn, country_sign2, month_term)
            oecd_growth = get_country_growth_pjt_byOECD(conn, country_sign2, month_term)
            print(imf_growth)
            print(oecd_growth)
            c_growth = (imf_growth + oecd_growth) / 2

        logger2.info(' ')        
        logger2.info(f'##### Country Growth: {c_growth} %')           
        logger2.info(f'##### Market Growth: {m_growth} %')
        logger2.info(f'##### Business Growth: {b_growth} %')        
        
        return (c_growth + m_growth*0.3 + b_growth*0.3*0.5)


'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    1. Economic Area
    '''
    # cli()
    # m1()
    # cpi()

    '''
    2. Market Area
    '''
    # cds()

    '''
    3. Business Area
    '''
    # container_Freight()

    '''
    4. Calculate Trend (Class)

    '''
    _trend = CalTrend()

    for nation in nations:
        if nation in ['EU', 'BR']:  # 몇 가지 정보가 존재하지 않아 제외
            continue
        trend = _trend.cal_trend(nation, 'stock', 'SPY', 0)     
        logger2.info(' ')
        logger2.info(f'##### {nation} total Trend: {trend} %')
        logger2.info(' ')