'''
Prgram 명: korea markets
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
목적: 대한민국 주식/채권/원자재/현금 데이터 분석
주요 내용
- 각 자산별 1년, 3년, 6년, 12년, 24년 마다의 평균값에서 +-시그마 범위를 벗어나는 경우 위험 표시
History
2023/11/14  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''
import pyminerva as mi
import pandas_ta as ta

from pykrx import stock
from pykrx import bond

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

TIMEFRAMES = ['1min', '1hour', '1day']

'''
1001 코스피
1028 코스피 200
1150 코스피 200 커뮤니케이션서비스
1151 코스피 200 건설
1152 코스피 200 중공업
1153 코스피 200 철강/소재
1154 코스피 200 에너지/화학
1155 코스피 200 정보기술
1156 코스피 200 금융
1157 코스피 200 생활소비재
1158 코스피 200 경기소비재
1159 코스피 200 산업재
1160 코스피 200 헬스케어
2001 코스닥
2002 코스닥 대형주
2203 코스닥 150
'''
sector_tickers = ['1001', '1028', '1150', '1151', '1152', '1153', '1154', '1155', '1156', '1157', '1158', \
           '1159', '1160', '2001', '2203']
core_tickers = ['1028', '2001'] # 코스피200, 코스닥
rep_tickers = ['KOSPI', 'KOSDAQ']  # 코스피, 코스닥
own_tickers = ['008770']  # 호텔신라, 

# today = to_date3
yesterday = (datetime.now() - timedelta(days=1)).date().strftime('%Y%m%d')

'''
1. Stocks
1.1 PER / PBR
'''
def per_pbr():

    df = stock.get_index_fundamental(yesterday, 'KOSPI')[:18]
    logger2.info(' KOSPI Fundamentals '.center(60, '*'))
    logger2.info(tabulate(df, headers='keys', tablefmt='rst', showindex=True))

    df = stock.get_index_fundamental(yesterday, 'KOSDAQ')[:10]
    logger2.info('')
    logger2.info(' KOSDAQ Fundamentals '.center(60, '*'))
    logger2.info(tabulate(df, headers='keys', tablefmt='rst', showindex=True))

    from_day = from_date_MT3
    plt.figure(figsize=(16, 4*len(core_tickers)))
    for i, ticker in enumerate(core_tickers):
        buf = stock.get_index_fundamental(from_day, to_date3, ticker)
        buf.dropna()
        plt.subplot(len(core_tickers), 1, i + 1)
        plt.grid()
        plt.plot(buf.index, buf['PER'], color='royalblue')
        if ticker == '1028':
            plt.title('KOSPI200 PER')
            plt.axhline(y=15, linestyle='--', color='red', linewidth=1, label='15X 과매도')
        else:
            plt.title('KOSDAQ PER')
            plt.axhline(y=30, linestyle='--', color='red', linewidth=1, label='30X 과매도')
        plt.xlabel('Date')
        plt.ylabel('PER')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/kr_m0110.png')

    plt.figure(figsize=(16, 4*len(core_tickers)))
    for i, ticker in enumerate(core_tickers):
        buf = stock.get_index_fundamental(from_day, to_date3, ticker)
        plt.subplot(len(core_tickers), 1, i + 1)
        plt.grid()
        plt.plot(buf.index, buf['PBR'], color='royalblue')
        
        if ticker == '1028':
            plt.title('KOSPI200 PBR')
            plt.axhline(y=0.8, linestyle='--', color='red', linewidth=1, label='0.8X 과매도')
        else:
            plt.title('KOSDAQ PBR')
            plt.axhline(y=1.5, linestyle='--', color='red', linewidth=1, label='1.5X 과매도')
        plt.xlabel('Date')
        plt.ylabel('PBR')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/kr_m0111.png')


'''
1.2 마켓별 공매도
'''
def short_selling():
    from_day = from_date_ST3
    # 공매도 건수
    plt.figure(figsize=(16, 4*len(rep_tickers)))
    for i, ticker in enumerate(rep_tickers):
        buf = stock.get_shorting_investor_volume_by_date(from_day, yesterday, ticker)
        plt.subplot(len(core_tickers), 1, i + 1)
        plt.plot(buf.index, buf['기관'], linewidth=1, label='기관', linestyle='-')
        plt.plot(buf.index, buf['개인'], linewidth=1, label='개인', linestyle='-')
        plt.plot(buf.index, buf['외국인'], linewidth=1, label='외국인', linestyle='-')
        plt.plot(buf.index, buf['합계'], linewidth=0.8, color='gray', label='합계', linestyle='--')    
        plt.grid()    
        if ticker == 'KOSPI':
            plt.title('KOSPI Short Selling Volume')
        elif ticker == 'KOSDAQ':
            plt.title('KOSDAQ Short Selling Volume')
        else:
            logger2.info('Short Selling parameter not found.')
        plt.xlabel('Date')
        plt.ylabel('공매도 건수')
        plt.legend()

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/kr_m0120.png')

    # 공매도 금액
    plt.figure(figsize=(16, 4*len(rep_tickers)))

    for i, ticker in enumerate(rep_tickers):
        buf = stock.get_shorting_investor_value_by_date(from_day, yesterday, ticker)
        plt.subplot(len(core_tickers), 1, i + 1)
        plt.plot(buf.index, buf['기관'], linewidth=1, label='기관', linestyle='-')
        plt.plot(buf.index, buf['개인'], linewidth=1, label='개인', linestyle='-')
        plt.plot(buf.index, buf['외국인'], linewidth=1, label='외국인', linestyle='-')
        plt.plot(buf.index, buf['합계'], linewidth=0.8, color='gray', label='합계', linestyle='--')    
        plt.grid()        
        if ticker == 'KOSPI':
            plt.title('KOSPI Short Selling 총금액(Amount)')
        elif ticker == 'KOSDAQ':
            plt.title('KOSDAQ Short Selling 총금액(Amount)')
        else:
            logger2.info('Short Selling parameter not found.')
        plt.xlabel('Date')
        plt.ylabel('공매도 금액')
        plt.legend()

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/kr_m0121.png')

    
'''
1.3 종목별 공매도 잔공
'''
def ticket_short_selling(own_tickers:list):
    from_day = from_date_ST3
    # 종목별 공매도 잔고
    plt.figure(figsize=(16, 5*len(own_tickers)))
    for i, ticker in enumerate(own_tickers):
        buf = stock.get_shorting_balance_by_date(from_day, yesterday, ticker)
        plt.subplot(len(core_tickers), 1, i + 1)
        plt.grid()
        plt.plot(buf.index, buf['공매도잔고'], label='공매도잔고', color='royalblue')
        name = stock.get_market_ticker_name(ticker)
        plt.title(f'{name} Balance')
        plt.xlabel('Date')
        plt.ylabel('공매도 잔고')
        plt.legend()

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/kr_m0130.png')


'''
2. Bonds
2.1 treasury yields
'''
def get_yields():
    df = bond.get_otc_treasury_yields(yesterday)
    logger2.info('##### treasury_yields #####')
    logger2.info(tabulate(df, headers='keys', tablefmt='rst', showindex=True))

    plt.figure(figsize=(16, 4*2))

    # 수익률 곡선
    plt.subplot(2, 1, 1)
    plt.title(f"국고채 수익률 곡선", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(df.index[:7], df['수익률'][:7], label='수익률 곡선', linewidth=1, color='blue', marker='o')
    plt.legend()

    # 국고채 10 vs 국고채 3년
    y10 = bond.get_otc_treasury_yields(from_date_ST, to_date3, "국고채10년")
    y3 = bond.get_otc_treasury_yields(from_date_ST, to_date3, "국고채3년")

    plt.subplot(2, 1, 2)
    plt.title(f"국고채 10년 vs 국고채 3년", fontdict={'fontsize':20, 'color':'g'})
    plt.grid()
    plt.plot(y10.index, y10['수익률'], label='국고채 10년', linewidth=1, color='royalblue', linestyle='--')
    plt.plot(y3.index, y3['수익률'], label='국고채 3년', linewidth=1, color='green', linestyle='--')
    plt.axhline(y=4.3, linestyle='--', color='blue', linewidth=1)
    plt.axhline(y=4.0, linestyle='--', color='green', linewidth=1)
    
    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/kr_m0210.png')


'''
3. Materials
'''


'''
4. Cash 
'''






'''
Main Fuction
'''

if __name__ == "__main__":

    for x in WATCH_TICKERS['KR']:

        for asset, tickers in x.items():

            for ticker in tickers:

                if ticker == '':
                    continue

                # settings.py 에서 get_stock_history_by_fmp with timeframe 파일 만들어 줌.
                logger2.info('')                
                logger2.info(f' ##### {ticker}')

                df = mi.get_stock_history_by_fmp(ticker, TIMEFRAMES)
                if df.empty:  # fmp 에서 읽지 못하면 다음에는 yfinance 에서 읽도록 보완함. 
                    logger2.error(f'{ticker} df by fmp is empty')

                    df = mi.get_stock_history_by_yfinance(ticker, TIMEFRAMES)
                    
                    logger2.info("dataframe from yfinance")
                    logger2.info(df.tail())
                
                mi.timing_strategy(ticker, 20, 200) # 200일 이평 vs 20일 이평

                mi.volatility_bollinger_strategy(ticker, TIMEFRAMES) # 임계값 찾는 Generic Algorithm 보완했음.

                mi.vb_genericAlgo_strategy(ticker, TIMEFRAMES) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm       

                mi.vb_genericAlgo_strategy2(ticker, TIMEFRAMES) # Bolinger Band Strategy + 임계값 찾는 Generic Algorithm           

                mi.reversal_strategy(ticker, TIMEFRAMES) 

                mi.trend_following_strategy(ticker, TIMEFRAMES)  # 단기 매매 아님. 중장기 매매 기법, 1day 데이터만으로 실행

                mi.control_chart_strategy(ticker)

                mi.gaSellHoldBuy_strategy(ticker)

                print('=== End ===')

    # 1. Stocks
    per_pbr()
    short_selling()
    ticket_short_selling(own_tickers) 

    # 2. Bonds
    get_yields()