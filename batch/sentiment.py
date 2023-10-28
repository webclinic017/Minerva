'''
program 명: sentiment.py
source Prgram 명: Cracks for Sentimentals
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
경제심리지수를 통한 위험 모니터링

History
20220816  Create
20220901  Naver Trend 추가
20220903  Google Trend 추가

'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *
from naverApi import *
from pytrends.request import TrendReq
import plotly.express as px
import json
import requests
from bs4 import BeautifulSoup as bs

# import logging, logging.config, logging.handlers

# ## Loads The Config File
# logging.config.fileConfig(batch_dir+'/logging.conf', disable_existing_loggers=False)

# # create a logger with the name from the config file. 
# # This logger now has StreamHandler with DEBUG Level and the specified format in the logging.conf file
# logger = logging.getLogger('batch')
# logger2 = logging.getLogger('report')

# 'application' code
logger.warning(sys.argv[0])
logger2.info(sys.argv[0])


###################################################################################################
# 모니터링 테이블 (Sent_Crack) 생성
###################################################################################################
# Connect DataBase
database = database_dir+'/Crack_Sent.db'
engine = 'sqlite:///' + database
conn = create_connection(database)

# 감성부문 Crack 집계 모니터링
def create_Sent_Crack(conn):
    with conn:
        cur=conn.cursor()
        cur.execute('create table if not exists Sent_Crack (Date text primary key, Tot_Percent real, \
            Tot_Count integer, CRSNT0001 integer, CRSNT0002 integer, CRSNT0003 integer, CRSNT0004 integer, \
            CRSNT0005 integer, CRSNT0006 integer, CRSNT0007 integer, CRSNT0008 integer, CRSNT0009 integer, \
            CRSNT0010 integer)')

    return conn

create_Sent_Crack(conn)
M_table = 'Sent_Crack'
M_query = f"SELECT * from {M_table}"
try:
    # 오늘자 Dataframe, db는 테이블에 있는 Dataframe 읽어온거.
    M_db = pd.read_sql_query(M_query, conn)
    buf = [today, 0,0,0,0,0,0,0,0,0,0,0,0]
    M_buffer = pd.DataFrame(data=[buf], columns=M_db.columns)
    logger2.info(M_db[-5:])
except Exception as e:
    print('Exception: {}'.format(e))


'''
1. 뉴스 인덱스와 S&P500 correlation 선행지수: CRSNT0001
https://fredblog.stlouisfed.org/2020/01/have-you-heard-the-news-news-affects-markets/?utm_source=series_page&utm_medium=related_content&utm_term=related_resources&utm_campaign=fredblog

'''
def news_sp500_corr():

    sp500 = fred.get_series(series_id='SP500', observation_start=from_date_LT)
    news = fred.get_series(series_id='STLENI', observation_start=from_date_LT)
    # display(news[-4:])

    fig, ax1 = plt.subplots(figsize=(15,5))
    ax1.plot(sp500, color='green', label='S&P 500')
    ax2 = ax1.twinx()
    ax2.plot(news, color='blue', label='News')
    plt.title(f"News vs SP500 Earning Correlation", fontdict={'fontsize':20, 'color':'g'})
    ax1.grid()
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    # plt.show()
    plt.savefig(reports_dir + "/sentiments_0010.png")

    # tp = turning_point_for_series(news)#분기단위 데이터라 .... TP 를 잡아내지못함.
    # tp_date = tp.index[-1] # 직전 TP 일자로 분석 
    tp_date = news.index[-4]#YoY 기준으로 일단.
    trend = trend_detector_for_series(news, tp_date)
    print('***** Trend Detector Index: ',trend)
    if trend < -15: 
        M_buffer['CRSNT0001'] += 1
    # display(M_buffer)



'''
2. Naver trend search: CRSNT0002
https://wooiljeong.github.io/python/pynaver/
https://wooiljeong.github.io/python/naver_datalab_open_api/
'''
def naver_trend_search():
    # from naverApi import *
    from datetime import date
    from dateutil.relativedelta import relativedelta

    # 지금은 가격이 아닌 일자별 검색량을 근거로 향후 트렌드를 예측하는 것이지만,
    # 이것을 가격 데이터를 넣으면....
    keyword_group_set = {
        'keyword_group_1': {'groupName': "1.STOCK", 'keywords': ["삼성전자","주가","코스피", "KOSPI"]},
        'keyword_group_2': {'groupName': "2.BOND", 'keywords': ["국공채","채권","국채", "10년물", "BOND"]},
        'keyword_group_3': {'groupName': "3.REAL ASSET", 'keywords': ["아파트","빌라","다세대"]},
        'keyword_group_4': {'groupName': "4.INFLATION", 'keywords': ["물가","인플레이션","휘발유"]},
        'keyword_group_5': {'groupName': "5.INVERSE", 'keywords': ["인버스","곱버스","하락베팅"]},
    }

    client_id = "FgRmyTtNtW_fX8vNKC3F"
    client_secret = "1p6jC1WBe5"

    from_date= str(date.today() - relativedelta(years = 3))
    to_date= today
    time_unit='date'
    device=''
    ages=[]
    gender=''

    naver = NaverDLabApi(client_id = client_id, client_secret = client_secret)
    naver.add_keyword_groups(keyword_group_set['keyword_group_1'])
    naver.add_keyword_groups(keyword_group_set['keyword_group_2'])
    naver.add_keyword_groups(keyword_group_set['keyword_group_3'])
    naver.add_keyword_groups(keyword_group_set['keyword_group_4'])
    naver.add_keyword_groups(keyword_group_set['keyword_group_5'])
    df = naver.get_data(from_date, to_date, time_unit, device, ages, gender)
    # display(df[-121::30])
    fig_1 = naver.plot_daily_trend()
    fig_1.savefig(reports_dir + "/sentiments_0020.png")

    if df['1.STOCK'][-61::15].any() > 5: # 세상 관심이 다 사라진 이후
        M_buffer['CRSNT0002'] += 1

    # fig_2 = naver.plot_monthly_trend()  # 오류확인필요 (20231022)
    fig_3 = naver.plot_pred_trend(days = 180)
    for i, f in enumerate(fig_3):
        f.savefig(reports_dir + f"/sentiments_003{i}.png")


'''
FRED Trend Terms
fred 에 트렌드 검색어는.
'''
def fred_trend():
    page = requests.get("https://fred.stlouisfed.org/")
    soup = bs(page.text, "html.parser")
    trend_fred = []
    elements = soup.find_all(class_='trending-search-item trending-search-gtm')
    for element in elements:
        print(element.text, ' >>>   ', end='')
        trend_fred.append(element.text)
        # sleep(1.5)  # display 모드에서 찬찬히 보라고.
    print('==== End ====')
    return trend_fred


'''
3. Google trend search
Google Trend 도 네이버와 같은 방식으로 구현하여 글로벌 트렌드 분석도 추가토록 함.
'''
def google_trend_search(keywords:list):
    kw_list=keywords[:5]
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(5,10))
    pytrends.build_payload(kw_list=kw_list, cat=0, timeframe='today 3-m', geo='', gprop='')
    try:
        df = pytrends.interest_over_time()
        df = df.reset_index()
        fig = px.line(df, x='date', y=kw_list, title='Keyword Web Search Interest Over Time')
        fig.write_image(reports_dir + "/sentiments_0040.png", width=1200, height=600)
    except Exception as e:
        print('Exception: {}'.format(e))
        logger.error(e)

    # https://github.com/GeneralMills/pytrends/pull/542: 현재 v5 버전에서 문제가 생겨 삭제되어버렸음. 후속 개선기능 아직 없음.
    # # warnings.filterwarnings('FutureWarning')
    # df_hourly = pytrends.get_historical_interest(kw_list, year_start=2022, month_start=1, day_start=1, hour_start=0, year_end=2022, month_end=9, day_end=3, hour_end=0, cat=0, geo='', gprop='', sleep=0)
    # df_hourly = df_hourly.reset_index()
    # fig = px.line(df_hourly, x="date", y=kw_list, title='Hourly Keyword Web Search Interest')
    # fig.write_image(reports_dir + "/sentiments_0050.png", width=1200, height=600)

    try:
        df = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
        buf = df.loc[['United States', 'South Korea', 'China', 'Germany', 'Japan', 'Austria', ]]
    except Exception as e:
        print('Exception: {}'.format(e))
        logger.error(e)


'''
4. 한국 경기심리지수(economic sentiment index) + 뉴스심리
기업과 소비자 모두를 포함한 민간의 경제상황에 대한 심리를 종합적으로 파악하기 위하여 BSI 및 CSI 지수를 합성하여 경제심리지수(ESI : Economic Sentiment Index)와 
뉴스기사 심리지수를 같이 분석함.
- 기업경기실사지수(BSI) : 기업가의 현재 경기수준에 대한 판단과 향후 전망 등을 설문조사를 통해 지수화 한 것
- 소비자동향지수(CSI) : 소비자들의 경기나 생활형편 등에 대한 주관적 판단과 전망, 미래 소비지출 계획 등을 설문조사를 통해 지수화 한 것
- 뉴스심리지수: 뉴스기사 텍스트 데이터를 이용하여 경제심리의 변화를 월별 경제심리지표 공표 이전에 신속하게 파악하여 경제동향 모니터링 및 정책수립을 위한 기초자료로 활용 
'''
def kor_esi_new_index():
# import datetime
    start_date = datetime.datetime.strptime(from_date_MT, '%d/%m/%Y').strftime('%Y%m')
    end_date   = datetime.datetime.strptime(to_date, '%d/%m/%Y').strftime('%Y%m')

    # 경제심리지수
    stat_code  = "513Y001"
    cycle_type = "M"
    item_1 = ['E1000', 'E2000']
    item_2 = []
    item_3 = []

    df = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df['TIME'] = df['TIME'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df['TIME'] = pd.to_datetime(df['TIME'], yearfirst=True)
    df['DATA_VALUE'] = (df['DATA_VALUE']).astype('float')

    df_esi_origin = df.loc[df['ITEM_CODE1'] == 'E1000']  # 경제심리지수 원계열
    df_esi_Coincident = df.loc[df['ITEM_CODE1'] == 'E2000']  # 경제심리지수 순환변동치

    buf = df_esi_Coincident[['TIME','DATA_VALUE']][-5:]

    # 뉴스심리지수
    stat_code  = "521Y001"
    cycle_type = "M"
    item_1 = ['A001']
    item_2 = []
    item_3 = []

    df2 = get_bok(bok_key, stat_code, cycle_type, start_date, end_date, item_1, item_2, item_3).drop(['ITEM_CODE4','ITEM_NAME4'], axis=1)
    df2.dropna(subset=['TIME','DATA_VALUE'], axis=0, inplace=True)
    df2['TIME'] = df2['TIME'].apply(lambda x: datetime.datetime.strptime(x, '%Y%m').strftime('%Y-%m-01'))
    df2['TIME'] = pd.to_datetime(df2['TIME'], yearfirst=True)
    df2['DATA_VALUE'] = (df2['DATA_VALUE']).astype('float')

    df2_news = df2.loc[df2['ITEM_CODE1'] == 'A001']  # 뉴스심리지수
    buf = df2_news[['TIME','DATA_VALUE']][-5:]

    # Graph
    plt.figure(figsize=(12,6))
    plt.title(f"economic sentiment index vs news sentiment index", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_esi_origin['TIME'], df_esi_origin['DATA_VALUE'], label='esi_origin', linewidth=0.5, color='gray')
    plt.plot(df_esi_Coincident['TIME'], df_esi_Coincident['DATA_VALUE'], label='esi_Coincident', linewidth=1, color='green')
    plt.plot(df2_news['TIME'], df2_news['DATA_VALUE'], label='news', linewidth=1, color='blue', marker='o')
    plt.legend()
    plt.savefig(reports_dir + "/sentiments_0050.png")

# '''
# 작업결과 Database insert
# '''
# def db_insert():
#     M_buffer['Tot_Count'] =  M_buffer.iloc[:, 3:].sum(axis=1)
#     M_buffer['Tot_Percent'] = M_buffer['Tot_Count']/(len(M_buffer.columns) - 3) * 100
#     try:
#         if M_db['Date'].str.contains(today).any():
#             buf = 'Duplicated: ' + M_db['Date']
#             logger.error(buf)
#             delete_Crack_By_Date(conn, 'Sent_Crack', date=today)
#         M_buffer.to_sql(M_table, con=engine, if_exists='append', chunksize=1000, index=False, method='multi')
#     except Exception as e:
#         print("################# Check Please: "+ e)
#     try:
#         # display(pd.read_sql_query(M_query, conn)[-5:])
#         buf = pd.read_sql_query(M_query, conn)[-5:]
#         logger2.info(buf)
#     except Exception as e:
#         print('################# Exception: {}'.format(e))

#     # 배치 프로그램 최종 종료시 Activate 후 실행
#     conn.close()



'''
Main Fuction
'''

if __name__ == "__main__":
    news_sp500_corr()
    naver_trend_search()
    keywords = fred_trend()
    logger2.info(keywords)
    google_trend_search(keywords)
    kor_esi_new_index()

    db_insert(M_db, M_table, M_query, M_buffer, conn, engine, logger, logger2)
