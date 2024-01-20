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
20240118  Table 방식 제거 
'''

# common area import
import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)
from settings import *

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# local import
from naverApi import *
from pytrends.request import TrendReq
import plotly.express as px
import json
import requests
from bs4 import BeautifulSoup as bs


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
    # keyword_group_set = {
    #     'keyword_group_1': {'groupName': "1.STOCK", 'keywords': ["삼성전자","주가","코스피", "KOSPI"]},
    #     'keyword_group_2': {'groupName': "2.BOND", 'keywords': ["국공채","채권","국채", "10년물", "BOND"]},
    #     'keyword_group_3': {'groupName': "3.REAL ASSET", 'keywords': ["아파트","빌라","다세대"]},
    #     'keyword_group_4': {'groupName': "4.INFLATION", 'keywords': ["물가","인플레이션","휘발유"]},
    #     'keyword_group_5': {'groupName': "5.INVERSE", 'keywords': ["인버스","곱버스","하락베팅"]},
    # }

    keyword_group_set = {
        'keyword_group_1': {'groupName': "1.코스피", 'keywords': ["삼성전자","주가","코스피", "KOSPI"]},
        'keyword_group_2': {'groupName': "2.코스닥", 'keywords': ["국공채","채권","국채", "10년물", "BOND"]},
        'keyword_group_3': {'groupName': "3.채권", 'keywords': ["아파트","빌라","다세대"]},
        'keyword_group_4': {'groupName': "4.부동산", 'keywords': ["물가","인플레이션","휘발유"]},
        'keyword_group_5': {'groupName': "5.인플레이션", 'keywords': ["인버스","곱버스","하락베팅"]},
    }    

    client_id = "FgRmyTtNtW_fX8vNKC3F"
    client_secret = "1p6jC1WBe5"

    from_date= str(date.today() - relativedelta(years = 3))
    to_date= to_date2
    time_unit='date'
    device=''
    ages=[]
    gender=''

    naver = NaverDLabApi(client_id = client_id, client_secret = client_secret)
    naver.add_keyword_groups(keyword_group_set['keyword_group_1'])
    # naver.add_keyword_groups(keyword_group_set['keyword_group_2'])
    # naver.add_keyword_groups(keyword_group_set['keyword_group_3'])
    # naver.add_keyword_groups(keyword_group_set['keyword_group_4'])
    # naver.add_keyword_groups(keyword_group_set['keyword_group_5'])
    df = naver.get_data(from_date, to_date, time_unit, device, ages, gender)
    # display(df[-121::30])
    fig_1 = naver.plot_daily_trend()
    fig_1.savefig(reports_dir + "/sentiments_0020.png")

    # if df['1.STOCK'][-61::15].any() > 5: # 세상 관심이 다 사라진 이후
    #     M_buffer['CRSNT0002'] += 1

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
    kw_list=keywords[:1]
    # pytrends = TrendReq(hl='en-US', tz=360, timeout=(5,10))
    pytrends = TrendReq(hl='en-US', tz=91, timeout=(10,25),retries=2, backoff_factor=0.1, requests_args={'verify':False})
    pytrends.build_payload(kw_list=kw_list, cat=0, timeframe='today 3-m', geo='', gprop='')
    try:
        df = pytrends.interest_over_time()
        df = df.reset_index()
        fig = px.line(df, x='date', y=kw_list, title='Keyword Web Search Interest Over Time')
        fig.write_image(reports_dir + "/sentiments_0040.png", width=1200, height=600)
    except Exception as e:
        print('Exception: {}'.format(e))
        logger.error(' >>> ' + e)

    # https://github.com/GeneralMills/pytrends/pull/542: 현재 v5 버전에서 문제가 생겨 삭제되어버렸음. 후속 개선기능 아직 없음.
    warnings.filterwarnings('FutureWarning')
    df_hourly = pytrends.get_historical_interest(kw_list, year_start=2022, month_start=1, day_start=1, hour_start=0, year_end=2022, month_end=9, day_end=3, hour_end=0, cat=0, geo='', gprop='', sleep=0)
    df_hourly = df_hourly.reset_index()
    fig = px.line(df_hourly, x="date", y=kw_list, title='Hourly Keyword Web Search Interest')
    fig.write_image(reports_dir + "/sentiments_0050.png", width=1200, height=600)

    try:
        df = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=False)
        buf = df.loc[['United States', 'South Korea', 'China', 'Germany', 'Japan', 'Austria', ]]
    except Exception as e:
        print('Exception: {}'.format(e))
        logger.error(' >>> ' + e)


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

    # 경기(경제) 심리지수
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
    plt.title(f"경기심리지수 vs 뉴스심리지수", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df_esi_origin['TIME'], df_esi_origin['DATA_VALUE'], label='경기심리지수(원계열)', linewidth=0.5, color='gray')
    plt.plot(df_esi_Coincident['TIME'], df_esi_Coincident['DATA_VALUE'], label='경기심리지수(순환변동치)', linewidth=1, color='green')
    plt.plot(df2_news['TIME'], df2_news['DATA_VALUE'], label='뉴스심리지수', linewidth=1, color='royalblue', marker='o')
    plt.legend()
    plt.savefig(reports_dir + "/sentiments_0050.png")


'''
Main Fuction
'''

if __name__ == "__main__":

    keywords = fred_trend()
    logger2.info(keywords)

    # google_trend_search(keywords)  # request full error 자주 발생하여...
    naver_trend_search()



    kor_esi_new_index()