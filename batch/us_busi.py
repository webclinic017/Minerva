'''
Prgram 명: 미국 자산별 Business부문의 technical Analysis 만들기
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
국가별 Economic 사이클을 분석하고, 자산시장별 금융환경 분석 그리고 ETF를 통한 섹터별 투자기회를 탐색하는 목적임.  
- country: 잠재성장률대비 real GDP YoY, (include KR.Export) = nominal GDP - CPI
- market: Nasdaq, S&P500, KOSPI, KOSDAQ, US 3Y/10Y/20Y BOND, KR 1Y BOND, GOLD, OIL, COLLAR, YEN, WON, EURO 
- business: 각 섹터별 ETF
History
- 20231204  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *
from global_ import CalcuTrend

import requests
import yfinance as yf
import pandas_ta as ta
from bs4 import BeautifulSoup as bs
from scipy import signal

'''
0. 공통영역 설정
'''
# logging
# logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
# logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

ta_list = ['mfi', 'ppo', 'bb', 'sma', 'rsi', 'vwap']


# Loading data, and split in train and test datasets
def get_data(ticker, window):

    ticker = yf.Ticker(ticker)
    df = ticker.history(period='36mo') # test: 10mo, real: 36mo
    df['feature'] = signal.detrend(df['Close'])
    df['mean'] = df['feature'].rolling(window=window).mean()    
    df['std'] = df['feature'].rolling(window=window).std()
    
    return df


def make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx):
    
    _df = df.copy() # 그래프만 그리면 되고, 본 df 에는 영향을 주지않기위해 복사본 만듬.
    _df = _df.dropna()
    
    # 상하위 값의 밴드를 가지고 있는 경우, 상하위 값을 벗어나면 탐색, trend 필요, trend 에 따라 밴드폭 조절
    if proj[0] == 'BULL':  # 확장 국면
        # 85% 분포를 갖는 시그마: 1.0364333894937898
        h_limit = _df[tech_type].mean() + (SIGMA_85 * _df[tech_type].std())
        l_limit = _df[tech_type].mean() - (SIGMA_85 * _df[tech_type].std())

        _df['pivot'] = np.where((_df[tech_type] > h_limit) & (_df[tech_type].shift(1) <= h_limit), 1, 0)
        _df['pivot'] = np.where((_df[tech_type] < l_limit) & (_df[tech_type].shift() >= l_limit), -1, _df['pivot'])         
        _1 = _df[_df['pivot'] == 1]
        _2 = _df[_df['pivot'] == -1]
        latest_date = max(_1.index[-1], _2.index[-1])
        if _1.empty and _2.empty:
            latest_date = '9999-99-99'
        elif _1.empty:
            latest_date = max(_2.index)
        elif _2.empty:
            latest_date = max(_1.index)
        else:
            latest_date = max(_1.index[-1], _2.index[-1])        

    elif proj[0] == 'BEAR':  # 수축 국면
        # 75% 분포를 갖는 시그마: 0.6744897501960817
        h_limit = _df[tech_type].mean() + (SIGMA_75 * _df[tech_type].std())
        l_limit = _df[tech_type].mean() - (SIGMA_75 * _df[tech_type].std())
        h_norm = _df[tech_type].mean() + _df[tech_type].std()
        l_norm = _df[tech_type].mean() - _df[tech_type].std()   

        _df['pivot'] = np.where((_df[tech_type] > h_limit) & (_df[tech_type].shift(1) <= h_limit), 1, 0)
        _df['pivot'] = np.where((_df[tech_type] < l_limit) & (_df[tech_type].shift() >= l_limit), -1, _df['pivot'])         
        _1 = _df[_df['pivot'] == 1]
        _2 = _df[_df['pivot'] == -1]

        if _1.empty and _2.empty:
            latest_date = '9999-99-99'
        elif _1.empty:
            latest_date = max(_2.index)
        elif _2.empty:
            latest_date = max(_1.index)
        else:
            latest_date = max(_1.index[-1], _2.index[-1])            

    elif proj[0] == 'STAY':  # 상하위 값이 없는 경우, 즉 threshold 를 crossover 만 탐색하는 경우 사용하는 루틴: trend 불필요 
        h_limit = 0
        l_limit = 0

        _df['pivot'] = np.where((_df[tech_type].shift(1)*_df[tech_type]) < 0, 1, 0)  # 1로 되는 일자부터 매수 또는 매도후 현금
        _df['pivot'] = np.where((_df[tech_type] < l_limit) & (_df[tech_type].shift() >= l_limit), -1, _df['pivot'])
        _1 = _df[_df['pivot'] == 1]
        _2 = _df[_df['pivot'] == -1]
        if _1.empty and _2.empty:
            latest_date = '9999-99-99'
        elif _1.empty:
            latest_date = max(_2.index)
        elif _2.empty:
            latest_date = max(_1.index)
        else:
            latest_date = max(_1.index[-1], _2.index[-1])

    else:  
        loger2.error('proj decision variable Not Found.')
        latest_date = '9999-99-99'


    plt.subplot(sbubplot_cnt, 1, idx)
    plt.plot(_df.index, _df[tech_type])

    max_val = max(_df[tech_type])
    min_val = min(_df[tech_type])
    if (max_val > 0) and (min_val < 0):       # 시각효과

        h_limit = 0
        l_limit = 0
        h_norm = 0
        l_norm = 0

        plt.axhline(y=h_limit, linestyle='--', color='red', linewidth=1)

        if _df.iloc[-1][tech_type].item() > 0:
            decision = 'Buy Signal'
        elif _df.iloc[-1][tech_type].item() < 0:
            decision = 'Sell Signal'
        else:
            decision = 'Stay'        
    else:
        plt.axhline(y=h_limit, linestyle='--', lw=1.2, color='red',)
        plt.axhline(y=l_limit, linestyle='--', lw=1.2, color='red',)  
        plt.axhline(y=h_norm, linestyle='--', lw=0.6, color='green',)
        plt.axhline(y=l_norm, linestyle='--', lw=0.6, color='green',)

        if _df.iloc[-1][tech_type].item() >= h_limit:
            decision = 'Sell Signal'
        elif _df.iloc[-1][tech_type].item() <= l_limit:
            decision = 'Buy Signal'
        else:
            decision = 'Stay'        

    plt.title(f"{ticker}: {tech_type} / {proj[0]}({round(proj[1],2)}%) / {decision}", fontdict={'fontsize':20, 'color':'g'})    

    plt.xlabel(f'Last Date: {latest_date}', loc='right')

    if latest_date != '9999-99-99':
        plt.plot(latest_date, _df[tech_type][latest_date], marker='o', color='red')

    
    return latest_date




'''
1. Technical Analysis
- Volume
    Money Flow Index (MFI)
    Accumulation/Distribution Index (ADI)
    On-Balance Volume (OBV)
    Chaikin Money Flow (CMF)
    Force Index (FI)
    Ease of Movement (EoM, EMV)
    Volume-price Trend (VPT)
    Negative Volume Index (NVI)
    Volume Weighted Average Price (VWAP)
    Volatility
    Average True Range (ATR)
    Bollinger Bands (BB)
    Keltner Channel (KC)
    Donchian Channel (DC)
    Ulcer Index (UI)

- Trend
    Simple Moving Average (SMA)
    Exponential Moving Average (EMA)
    Weighted Moving Average (WMA)
    Moving Average Convergence Divergence (MACD)
    Average Directional Movement Index (ADX)
    Vortex Indicator (VI)
    Trix (TRIX)
    Mass Index (MI)
    Commodity Channel Index (CCI)
    Detrended Price Oscillator (DPO)
    KST Oscillator (KST)
    Ichimoku Kinkō Hyō (Ichimoku)
    Parabolic Stop And Reverse (Parabolic SAR)
    Schaff Trend Cycle (STC)

- Momentum
    Relative Strength Index (RSI)
    Stochastic RSI (SRSI)
    True strength index (TSI)
    Ultimate Oscillator (UO)
    Stochastic Oscillator (SR)
    Williams %R (WR)
    Awesome Oscillator (AO)
    Kaufman’s Adaptive Moving Average (KAMA)
    Rate of Change (ROC)
    Percentage Price Oscillator (PPO)
    Percentage Volume Oscillator (PVO)
'''
# Volume.Money Flow Index (MFI)
def mfi(ticker, df, proj, tech_type, sbubplot_cnt, idx):
    df[tech_type] = df.ta.mfi(close=df['Close'],length=14)
    latest_date = make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx)

    return df, latest_date

# Momentum.Percentage Price Oscillator (PPO)
def ppo(ticker, df, proj, tech_type, sbubplot_cnt, idx):    
    buf= df.ta.ppo(close=df['Close'], append=False)
    df[tech_type] = buf['PPO_12_26_9']
    latest_date = make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx)
                    
    return df, latest_date

# Volatility.Bollinger Bands (BB)
def bb(ticker, df, proj, tech_type, sbubplot_cnt, idx):    
    
    buf = df.ta.bbands(close=df['Close'], length=20, append=False)
    buf['high_limit'] = buf['BBU_20_2.0'] + (buf['BBU_20_2.0'] - buf['BBL_20_2.0']) / 2
    buf['low_limit'] = buf['BBL_20_2.0'] - (buf['BBU_20_2.0'] - buf['BBL_20_2.0']) / 2    
    df[tech_type] = (df['Close'] - buf['low_limit']) / (buf['high_limit'] - buf['low_limit'])
    latest_date = make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx)    
    
    return df, latest_date


# Trend.Simple Moving Average (SMA)
def sma(ticker, df, proj, tech_type, sbubplot_cnt, idx):
    
    df[tech_type] = df.ta.sma(20) - df.ta.sma(200)
    slope = 'NA'
    latest_date = make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx)
    
    return df, latest_date


# Momentum.Relative Strength Index (RSI)
def rsi(ticker, df, proj, tech_type, sbubplot_cnt, idx):    
    df[tech_type] = df.ta.rsi(14)
    latest_date = make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx)
    
    return df, latest_date


# Volume.Volume Weighted Average Price (VWAP)
def vwap(ticker, df, proj, tech_type, sbubplot_cnt, idx):    
    df[tech_type] = df.ta.vwap(15)
    latest_date = make_plot(ticker, df, proj, tech_type, sbubplot_cnt, idx)
    
    return df, latest_date






'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    0. 공통
    '''
    # get_tickers()



    '''
    1. Technical Analysis
    '''

    _trend = CalcuTrend() # 다음에는 Alpha 테이블에서 읽어온 것으로 처리하도록 변경해야함.

    
    for x in WATCH_TICKERS['US']:

        for key, item in x.items():

            for ticker in item:
                nation = 'US'
                asset = key
                if ticker == '':
                    df = pd.DataFrame()
                    continue                

                trend_now, x,y,z = _trend.cal_trend(nation, asset, ticker, 0)
                trend_6mo, x,y,z = _trend.cal_trend(nation, asset, ticker, 6)
                diff = trend_now - trend_6mo
                size = diff / trend_now * 100
                proj = []
                if diff > 0:  # 수축 국면
                    proj = ['BEAR', size]
                elif  diff == 0:  # 유지 국면
                    proj = ['STAY', 0]
                else:
                    proj = ['BULL', size]
                logger2.info('')
                logger2.info(f'##### {nation} Trend Now: {trend_now} %')
                logger2.info(f'##### {nation} Trend after 6 month: {trend_6mo} %')
                logger2.info(f'##### {nation} Economic Trend: {proj[0]} {str(round(proj[1], 2))}%')        
            
                sbubplot_cnt = len(ta_list)

                plt.figure(figsize=(16, 4*sbubplot_cnt))

                df = get_data(ticker, 20)
                # Volume.Money Flow Index (MFI)
                tech_type = 'mfi'
                df, latest_date = mfi(ticker, df, proj, tech_type, sbubplot_cnt, 1)


                # Momentum.Percentage Price Oscillator (PPO)
                tech_type = 'ppo'
                df, latest_date = ppo(ticker, df, proj, tech_type, sbubplot_cnt, 2)


                # Volatility.Bollinger Bands (BB)
                tech_type = 'bb'
                df, latest_date = bb(ticker, df, proj, tech_type, sbubplot_cnt, 3)


                # Trend.Simple Moving Average (SMA)
                tech_type = 'sma'
                df, latest_date = sma(ticker, df, proj, tech_type, sbubplot_cnt, 4)      


                # Momentum.Relative Strength Index (RSI)
                tech_type = 'rsi'
                df, latest_date = rsi(ticker, df, proj, tech_type, sbubplot_cnt, 5)      
  

                # Volume.Volume Weighted Average Price (VWAP)
                tech_type = 'vwap'








                plt.tight_layout()  # 서브플롯 간 간격 조절
                plt.savefig(reports_dir + f'/us_b0100_{ticker}.png')






    '''
    2. Fundermental Analysis
    이것도 그래프로. (bar plot)
    '''


    '''
    3. Report
    '''
