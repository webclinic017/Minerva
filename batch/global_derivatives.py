'''
Prgram 명: 미국시장부문의 파생상품 동향 분석용
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
다양한 상품군들에 대한 파생(선물/옵션)등의 동향을 분석하여
- 현물시장의 투자정보 제공
- 현물시장 대량 거래시 헷징을 위한 근거

History
2023/12/24  Creat
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''
import yfinance as yf
import pandas_ta as ta
import pygad
import pygad.kerasga
import gym

from scipy import signal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gym import spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from math import sqrt, exp

from yahoo_fin import stock_info as si
from dateutil import parser
from yahoo_fin import options
import yfinance as yf

# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))


Symbols = ['^SPX', '^NDX']

# Commitment of Traders (COT) report 참조
# symbol: 파생상품의 종류
# ticker: 현물시장에서 상품종류
COT_SYMBOLS = [
               {'symbol':'NQ', 'ticker':'QQQ', 'name':'Nasdaq 100 E-Mini', 'contract_units': 'NASDAQ 100 STOCK INDEX x $2'},
               {'symbol':'ES', 'ticker':'SPY', 'name':'S&P 500 E-Mini (ES)', 'contract_units': '$50 X S&P 500 INDEX'}, 
               {'symbol':'YM', 'ticker':'DIA', 'name':'Dow Industry 30 E-Mini (YM)', 'contract_units': '$5 X DJIA INDEX'},
               {'symbol':'QR', 'ticker':'IWM', 'name':'Russell 2000 E-Mini (QR)', 'contract_units': 'RUSSEL 2000 INDEX X $50'},
               {'symbol':'EW', 'ticker':'IJH', 'name':'S&P Midcap E-Mini (EW)', 'contract_units': 'S&P 400 INDEX X $100'},
               {'symbol':'VI', 'ticker':'VIXY', 'name':'S&P 500 VIX (VI)', 'contract_units': '$5 X S&P 500 INDEX'},

               {'symbol':'ZN', 'ticker':'IEF', 'name':'10-Year T-Note (ZN)', 'contract_units': 'CONTRACTS OF $100,000 FACE VALUE'},
               {'symbol':'ZB', 'ticker':'TLT', 'name':'30-Year T-Bond (ZB)', 'contract_units': 'CONTRACTS OF $100,000 FACE VALUE'},

               {'symbol':'BT', 'ticker':'BITO', 'name':'Bitcoin CME Futures (BT)', 'contract_units': '5 Bitcoins'},  # 현물 etf 없음. 대체사용

               {'symbol':'DX', 'ticker':'UUP', 'name':'US Dollar Index', 'contract_units': 'U.S. DOLLAR INDEX X $1000'}, 
               {'symbol':'E6', 'ticker':'FXE', 'name':'Euro FX (E6)', 'contract_units': 'CONTRACTS OF 125,000 EUROS'},                
               {'symbol':'J6', 'ticker':'FXY', 'name':'Japanese Yen (J6)', 'contract_units': 'CONTRACTS OF JPY 12,500,000'},
               {'symbol':'B6', 'ticker':'FXB', 'name':'British Pound (B6)', 'contract_units': 'CONTRACTS OF GBP 62,500'},                
               {'symbol':'S6', 'ticker':'FXF', 'name':'Swiss Franc (S6)', 'contract_units': 'CONTRACTS OF 125,000 SWISS FRANCS'},
               {'symbol':'M6', 'ticker':'MXN', 'name':'Mexican Peso (M6)', 'contract_units': 'CONTRACTS OF 500,000 MEXICAN PESOS'},
               {'symbol':'L6', 'ticker':'EWZ', 'name':'Brazilian Real (L6)', 'contract_units': 'CONTRACTS OF BRL 100,000'}, # 화폐 etf 는 없음. 대체사용

               {'symbol':'GC', 'ticker':'GLD', 'name':'Gold (GC)', 'contract_units': 'CONTRACTS OF 100 TROY OUNCE'},
               {'symbol':'HG', 'ticker':'CPER', 'name':'High Grade Copper (HG)', 'contract_units': ''},               
               {'symbol':'CL', 'ticker':'UCO', 'name':'Crude Oil', 'contract_units': 'CONTRACTS OF 1,000 BARRELS'},
               {'symbol':'BZ', 'ticker':'BNO', 'name':'Crude Oil BRENT (BZ)', 'contract_units': ''},               
               {'symbol':'RB', 'ticker':'UGA', 'name':'Gasoline RBOB (RB)', 'contract_units': 'CONTRACTS OF 42,000 U.S. GALLONS'},
               {'symbol':'NG', 'ticker':'UNG', 'name':'Natural Gas', 'contract_units': ''},
               ]
               

'''
1. Future
1.1 The Commitment of Traders (COT) Report from financial modeling.com API
- Commitment of Traders (COT) 리포트는 선물 및 옵션 시장에서 다양한 거래자 그룹의 포지션 정보를 담고 있는 주간 보고서입니다. 
  이 보고서는 미국 선물거래위원회(CFTC)에서 발표되며, 선물 시장 참가자들이 보유한 계약 수량을 나누어 상업적 거래자, 비상업적 거래자, 
  그리고 비보고 거래자로 구분하여 제공합니다.
- 상업적 거래자 (Commercial Traders): 이 그룹은 실제 상품을 생산하거나 소비하는 기업들을 포함합니다. 예를 들어 곡물 생산자나 에너지 기업 등이 여기에 속합니다. 
  이들은 시장에서 발생하는 가격의 변동에 대비하기 위해 선물 계약을 사용합니다. 
  COT 리포트에서는 주로 헤지를 목적으로 선물 계약을 보유하는 상업적 거래자의 포지션 동향을 확인할 수 있습니다.
- 비상업적 거래자 (Non-Commercial Traders 또는 큰 투기자들): 주로 투기적 목적으로 선물 시장에 참여하는 거래자들로, 주식 펀드, 헤지 펀드, 자금 관리 회사 등이 
  여기에 속합니다. 이 그룹은 주로 차트, 기술 분석, 투기적 거래 등을 통해 이익을 추구하며, 그들의 대규모 포지션 변동은 시장의 향방을 예측하는 데 활용될 수 있습니다.
- 비보고 거래자 (Non-Reportable Positions): CFTC에 보고하지 않는 작은 규모의 거래자들을 나타냅니다. 
  이 그룹은 상대적으로 작은 규모의 포지션을 보유하고 있으며, 주로 소규모 투자자나 소매 트레이더들이 여기에 해당됩니다.
- COT 리포트는 이러한 다양한 거래자 그룹의 포지션 동향을 통해 시장 참가자들이 어떻게 행동하고 있는지를 추적할 수 있습니다. 
  큰 투기자들의 대규모 거래 포지션 변동은 종종 시장의 향방 전환을 시사할 수 있습니다. 
  이 보고서는 특히 기술적 분석과 결합하여 트레이더들이 시장 동향을 분석하고 전략을 수립하는 데 도움을 줍니다.

1.1.1 US
1.1.2 KR
1.1.3 DE
1.1.4 JP
1.1.5 CN
'''
def COT_analyse():
    df = get_cot_analysis_by_dates()
    df = df.sort_values(['symbol','date'], ascending=False).reset_index(drop=True)
    # 전체 그림의 크기를 설정
    plt.figure(figsize=(10, 4*len(COT_SYMBOLS)))
    for i, cot in enumerate(COT_SYMBOLS):
        result = df[df['symbol'].str.contains(cot['symbol'], case=False, na=False)]
        result['date'] = result['date'].apply(parse_date)
        if result.empty:
            continue
        name = result.iloc[0]['name']
        situation = result.iloc[0]['marketSituation']
        sentiment = result.iloc[0]['marketSentiment']
        netPostion = result.iloc[0]['netPostion']
        previousNetPosition = result.iloc[0]['previousNetPosition']

        if result.iloc[0]['reversalTrend'] == True:
            reversalTrend = 'Reversal Trend YES'
        else:
            reversalTrend = 'Keep Staying'
        plt.subplot(len(COT_SYMBOLS), 1, i + 1)
        plt.title(name+': '+str(previousNetPosition)+ ' -> ' +str(netPostion)+' / '+situation+' / '+sentiment+' / '+reversalTrend)
        plt.plot(result['date'], result['currentLongMarketSituation'], color='royalblue', label='current long')
        plt.plot(result['date'], result['currentShortMarketSituation'], color='red', label='current short')
        plt.xlabel('date')
        plt.ylabel('percentage')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/global_d0110.png')


def COT_report():
    buf = get_cot_report_by_dates()
    max_date = max(buf['date'])
    max_date = pd.to_datetime(max_date).date()
    df = buf[buf['date'].str.contains(str(max_date), case=False, na=False)]
    df = df.sort_values(['symbol','date'], ascending=False).reset_index(drop=True)

    logger2.info(f' ################ COT recent Date: {max_date}')

    # 전체 그림의 크기를 설정
    plt.figure(figsize=(10, 3*len(COT_SYMBOLS)))
    for i, cot in enumerate(COT_SYMBOLS):
        result = df[df['symbol'].str.contains(cot['symbol'], case=False, na=False)]
        result['date'] = result['date'].apply(parse_date)
        name = result.iloc[0]['short_name']

        open_interest_all = result.iloc[0]['open_interest_all']
        noncomm_positions_long_all = result.iloc[0]['noncomm_positions_long_all']
        noncomm_positions_short_all = result.iloc[0]['noncomm_positions_short_all']
        comm_positions_long_all = result.iloc[0]['comm_positions_long_all']
        comm_positions_short_all = result.iloc[0]['comm_positions_short_all']

        change_in_open_interest_all = result.iloc[0]['change_in_open_interest_all']
        change_in_noncomm_long_all = result.iloc[0]['change_in_noncomm_long_all']
        change_in_noncomm_short_all = result.iloc[0]['change_in_noncomm_short_all']
        change_in_comm_long_all = result.iloc[0]['change_in_comm_long_all']
        change_in_comm_short_all = result.iloc[0]['change_in_comm_short_all']

        categories = [
                    'hedge long', 'hedge short', 'dealer long', 'dealer short',
                      ]
        categories2 = [
                    'hedge long*', 'hedge short*', 'dealer long*', 'dealer short*']        
        values = [
            noncomm_positions_long_all,
            noncomm_positions_short_all,
            comm_positions_long_all,
            comm_positions_short_all,
            ]
        
        values2 = [
            change_in_noncomm_long_all,
            change_in_noncomm_short_all,
            change_in_comm_long_all,
            change_in_comm_short_all
            ]        

        plt.subplot(len(COT_SYMBOLS), 1, i + 1)
        plt.title(name+'/ contracts: '+str(open_interest_all)+'/ changes: '+str(change_in_open_interest_all))
        plt.bar(categories, values, color='royalblue', label='contracts')
        plt.bar(categories2, values2, color='orange', label='Changes')        
        plt.ylabel('count')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/global_d0120.png')






'''
1.2 The Commitment of Traders (COT) Report Summary Analyse from COT 선물거래소 다운로드 데이터(pct data)
COT 보고서의 각 trader 별 베팅 비율을 근간으로 BB Strategy 투자분석
https://wire.insiderfinance.io/download-sentiment-data-for-financial-trading-with-python-b07a35752b57
1) Insight into Market Sentiment
2) Early Warning Signals
3) Confirmation of Technical Analysis
4) Risk Management
5) Long-Term Investment Insights
6) Data-Driven Trading
'''
class OTCBBstg():

    def __init__(self):
        pass

    def set_cot_file(self, symbol):
        df = get_cot_report_by_symbol(symbol)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date']).reset_index(drop=True)
        df = df.drop_duplicates()
        df.to_csv(data_dir + f'/cot_{symbol}.csv', index=False)

    def __get_cot_data(self, cot_file, ticker_file, ticker):
            # Read Sentiment Data
            df_sd = pd.read_csv(cot_file)
            df_sd = df_sd.reset_index()
            df_sd['date'] = pd.to_datetime (df_sd['date'])

            # Read / Get & Save Market Data
            ticker = yf.Ticker(ticker)
            df = ticker.history(
                interval='1d',
                start=min(df_sd['date']),
                end=max(df_sd['date']))
            
            if len(df) <= 0:
                logger2.info(f'ticker not found: {ticker}')
                return None
            
            df = df.reset_index()
            df['Date'] = df['Date'].dt.date
            df = df[['Date','Close']]
            df.columns = ['date', 'close']
            if len(df) > 0: df.to_csv(ticker_file, index=False)

            df = pd.read_csv(ticker_file)
            df['date'] = pd.to_datetime(df['date'])
            # Merge Market Sentiment Data And Market Data
            tolerance = pd.Timedelta('7 day')

            df = pd.merge_asof(left=df_sd,right=df,left_on='date',right_on='date',direction='backward',tolerance=tolerance)
            # Clean Data And Rename Columns
            df = df.dropna()

            return df

    def __process(self, df, field, bb_length, min_bandwidth, max_buy_pct, min_sell_pct, CASH):
        # Generate a copy to avoid changing the original data
        df = df.copy().reset_index(drop=True)
        # Calculate Bollinger Bands With The Specified Field
        df.ta.bbands(close=df[field], length=bb_length, append=True)
        df['high_limit'] = df[f'BBU_{bb_length}_2.0'] + (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0']) / 2
        df['low_limit'] = df[f'BBL_{bb_length}_2.0'] - (df[f'BBU_{bb_length}_2.0'] - df[f'BBL_{bb_length}_2.0']) / 2
        df['close_percentage'] = np.clip((df[field] - df['low_limit']) / (df['high_limit'] - df['low_limit']), 0, 1)
        df['bandwidth'] = np.clip(df[f'BBB_{bb_length}_2.0'] / 100, 0, 1)
        df = df.dropna()
        # Buy Signal
        df['signal'] = np.where((df['bandwidth'] > min_bandwidth) & (df['close_percentage'] < max_buy_pct), 1, 0)
        # Sell Signal
        df['signal'] = np.where((df['close_percentage'] > min_sell_pct), -1, df['signal'])
        # Remove all rows without operations, rows with the same consecutive operation, first row selling, and last row buying
        result = df[df['signal'] != 0]
        result = result[result['signal'] != result['signal'].shift()]
        if (len(result) > 0) and (result.iat[0, -1] == -1): result = result.iloc[1:]
        if (len(result) > 0) and (result.iat[-1, -1] == 1): result = result.iloc[:-1]
        # Calculate the reward / operation
        result['total_reward'] = np.where(result['signal'] == -1, (result['close'] - result['close'].shift()) * (CASH // result['close'].shift()), 0)
        # Generate the result
        total_reward = result['total_reward'].sum()
        wins = len(result[result['total_reward'] > 0])
        losses = len(result[result['total_reward'] < 0])

        return total_reward, wins, losses


    def cot_bb_stg_report(self, symbol, ticker):
        # Configuration
        np.set_printoptions(suppress=True)
        pd.options.mode.chained_assignment = None

        cot_file = data_dir + f'/cot_{symbol}.csv'
        symbol_file = data_dir + f'/{ticker}_{symbol}.csv'
        CASH = 10_000
        BB_LENGTH = 20
        MIN_BANDWIDTH = 0
        MAX_BUY_PCT = 0.25
        MIN_SELL_PCT = 0.75

        # Get Required Data
        df = self.__get_cot_data(cot_file, symbol_file, ticker)

        # Get Result Based Calculating the BB on Each Field to Check Which is the Most Accurate
        for field in ['comm_positions_long_all', 'comm_positions_short_all', 'noncomm_positions_long_all', 'noncomm_positions_short_all']:
            total_reward, wins, losses = self.__process(df, field, BB_LENGTH, MIN_BANDWIDTH, MAX_BUY_PCT, MIN_SELL_PCT, CASH)
            logger2.info(f' Result of {ticker} for (Field: {field}) '.center(60, '*'))
            logger2.info(f"* Profit / Loss           : {total_reward:.2f}")
            logger2.info(f"* Wins / Losses           : {wins} / {losses}")
            logger2.info(f"* Win Rate (BB length=20) : {(100 * (wins/(wins + losses)) if wins + losses > 0 else 0):.2f}%")



'''
3.1 Symbols(^SPX, ^NDX) Future during all times
'''
def option_chain(Symbols):

    plt.figure(figsize=(18, 4*len(Symbols)))

    for i, symbol in enumerate(Symbols):
        opt_exps = options.get_expiration_dates(symbol)
        # logger2.debug(opt_exps)

        chain = options.get_options_chain(symbol)
        logger2.info(chain['calls'].head())
        logger2.info(chain['puts'].head())
        df_call=pd.DataFrame(chain['calls'])
        df_put=pd.DataFrame(chain['puts'])

        df = yf.Ticker(symbol)
        curr_price = df.fast_info['previousClose']
        df_call['Strike'].dropna(axis=0, inplace=True)
        df_call['Strike'] = df_call['Strike'].astype('float')
        df_call['Volume'].dropna(axis=0, inplace=True)
        df_call['Volume'].replace('-', 0, inplace=True)
        df_call['Volume'] = df_call['Volume'].astype('float')

        df_put['Strike'].dropna(axis=0, inplace=True)
        df_put['Strike'] = df_put['Strike'].astype('float')
        df_put['Volume'].dropna(axis=0, inplace=True)
        df_put['Volume'].replace('-', 0, inplace=True)
        df_put['Volume'] = df_put['Volume'].astype('float')

        plt.subplot(len(Symbols), 1, i + 1)  
        plt.title(f'{symbol} Option all the time')
        plt.bar(x=df_call['Strike'], height=df_call['Volume'], color='orange', label='Call', width=3.5)
        plt.bar(x=df_put['Strike'], height=df_put['Volume'], color='royalblue', label='Put', width=3.5)
        plt.axvline(x=curr_price, color='green', linestyle=':', linewidth=2, label=curr_price)
        plt.legend()
        plt.grid()

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/global_d0310.png')    


'''
3.2 Symbols(^SPX, ^NDX): Today's Options Chain for SPY Future
'''
def option_chain_today(Symbols):
    
    def get_option_sp500(x):
        if (x.find('SPXW') < 0): #SPX
            option = x[9]
        elif (x.find('SPXW') == 0): #SPXW
            option = x[10]
        else:
            return -1    
        return option

    def get_expdate_sp500(x):
        if (x.find('SPXW') < 0):
            _ = '20'+x[3:9]
            exp_date = pd.to_datetime(_, format='%Y%m%d', yearfirst=True).strftime('%Y-%m-%d')
        elif (x.find('SPXW') == 0):
            _ = '20'+x[4:10]
            exp_date = pd.to_datetime(_, format='%Y%m%d', yearfirst=True).strftime('%Y-%m-%d')
        else:
            return -1    
        return exp_date
    
    def get_option_nasdaq100(x):
        if (x.find('NDXP') < 0): #NDX
            option = x[9]
        elif (x.find('NDXP') == 0): #NDXP
            option = x[10]
        else:
            return -1    
        return option

    def get_expdate_nasdaq100(x):
        if (x.find('NDXP') < 0): #NDX
            _ = '20'+x[3:9]
            exp_date = pd.to_datetime(_, format='%Y%m%d', yearfirst=True).strftime('%Y-%m-%d')
        elif (x.find('NDXP') == 0): #NDXP
            _ = '20'+x[4:10]
            exp_date = pd.to_datetime(_, format='%Y%m%d', yearfirst=True).strftime('%Y-%m-%d')
        else:
            return -1    
        return exp_date
    
    plt.figure(figsize=(36, 4*len(Symbols)))  # *2 는 bar chart 를 call 과 put 을 각각 먼저 그려보기 위함임. 어떤 것이 먼저 덮을지 ....
    for i, symbol in enumerate(Symbols):
        df = yf.Ticker(symbol)
        df_exp = df.options
        buf = pd.DataFrame()
        for j, date in enumerate(df_exp):
        #     print(date, ' >>> ', end='')
            opt = df.option_chain(date)
            for x in opt:
                try:
                    if symbol == '^SPX':
                        temp = x[x['volume'] > 3000.0]
                    else: # '^NDX'
                        temp = x[x['volume'] > 100.0]
                except:
                    if j == 0:
                        # logger2.info(x)
                        logger2.info('=====' + x['shortName'])
                        logger2.info(f"regularMarketPrice: {x['regularMarketPrice']:,}")
                        logger2.info(f"regularMarketPreviousClose: {x['regularMarketPreviousClose']:,}")
                        logger2.info(f"regularMarketChange: {x['regularMarketChange']:,}")
                        logger2.info(f"regularMarketChangePercent: {x['regularMarketChangePercent']:,}")

                        logger2.info(f"regularMarketVolume: {x['regularMarketVolume']:,}")
                        logger2.info(f"averageDailyVolume10Day: {x['averageDailyVolume10Day']:,}")
                        logger2.info(f"averageDailyVolume3Month: {x['averageDailyVolume3Month']:,}")
                        logger2.info(f"twoHundredDayAverage: {x['twoHundredDayAverage']:,}")
                    
                        logger2.info(f"fiftyDayAverage: {x['fiftyDayAverage']:,}")
                        logger2.info(f"fiftyDayAverageChange: {x['fiftyDayAverageChange']:,}")
                        logger2.info(f"fiftyDayAverageChangePercent: {x['fiftyDayAverageChangePercent']:,}")

                        logger2.info(f"twoHundredDayAverage: {x['twoHundredDayAverage']:,}")                        
                        logger2.info(f"twoHundredDayAverageChange: {x['twoHundredDayAverageChange']:,}")
                        logger2.info(f"twoHundredDayAverageChangePercent: {x['twoHundredDayAverageChangePercent']:,}")

                        logger2.info(f"fiftyTwoWeekRange: {x['fiftyTwoWeekRange']}")
                        logger2.info(f"fiftyTwoWeekLowChange: {x['fiftyTwoWeekLowChange']:,}")
                        logger2.info(f"fiftyTwoWeekHighChangePercent: {x['fiftyTwoWeekHighChangePercent']:,}")

                if len(temp) > 0:
                    # logger2.info(str(temp['contractSymbol'].values))
                    # logger2.info(str(temp['volume'].values))                    
                    buf = pd.concat([buf, temp], axis=0)
        print('====== End ======')        

        if symbol == '^SPX':
            buf['Option'] = buf['contractSymbol'].apply(get_option_sp500)
            buf['Exp_date'] = buf['contractSymbol'].apply(get_expdate_sp500)
        elif symbol == '^NDX':
            buf['Option'] = buf['contractSymbol'].apply(get_option_nasdaq100)
            buf['Exp_date'] = buf['contractSymbol'].apply(get_expdate_nasdaq100)
        else:
            logger.error(f'Symbol not found: {symbol}')

        buf_pivot = pd.pivot_table(buf, index=['Option','Exp_date'], values=['volume'], aggfunc=np.sum, fill_value=0)
        # logger2.info(buf_pivot)

        buf_pivot_call = buf_pivot.iloc[buf_pivot.index.get_level_values('Option') == 'C']
        buf_pivot_call.reset_index(level=0, drop=False, inplace=True)
        buf_pivot_put = buf_pivot.iloc[buf_pivot.index.get_level_values('Option') == 'P']
        buf_pivot_put.reset_index(level=0, drop=False, inplace=True)
        buf_pivot_put.sort_index(ascending=True, inplace=True)

        # buf_pivot_call.index = pd.to_datetime(buf_pivot_call.index)
        # buf_pivot_put.index = pd.to_datetime(buf_pivot_put.index)

        plt.subplot(len(Symbols), 2, i*2 + 1)
        plt.title(f"Today's Options Chain for {symbol}", fontdict={'fontsize':20, 'color':'g'})
        plt.bar(x=buf_pivot_call.index, height=buf_pivot_call.volume, label='Call', color='orange')
        plt.bar(x=buf_pivot_put.index, height=buf_pivot_put.volume, label='Put', color='royalblue')
        plt.grid()
        plt.legend()

        plt.subplot(len(Symbols), 2, i*2 + 2)
        plt.title(f"Today's Options Chain for {symbol}", fontdict={'fontsize':20, 'color':'g'})
        plt.bar(x=buf_pivot_put.index, height=buf_pivot_put.volume, label='Put', color='royalblue')        
        plt.bar(x=buf_pivot_call.index, height=buf_pivot_call.volume, label='Call', color='orange')
        plt.grid()
        plt.legend()        

    plt.tight_layout()  # 서브플롯 간 간격 조절
    plt.savefig(reports_dir + '/global_d0320.png')









'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    0. 공통 영역
    '''

    '''
    1. COT 분석과 리포트: Commitment of Traders (COT) 주간단위 Report
    '''
    COT_analyse()
    COT_report()


    # '''
    # 2. COT BB Strategy 분석
    # '''
    # oct_bb = OTCBBstg()

    # for cot in COT_SYMBOLS:
    #     symbol = cot['symbol']
    #     ticker = cot['ticker']
    #     name = cot['name']
    #     oct_bb.set_cot_file(symbol)

    #     logger2.info(f''.center(60, ' '))        
    #     logger2.info(f' {name} '.center(60, '#'))
    #     oct_bb.cot_bb_stg_report(symbol, ticker)

    # '''
    # 3. Option Chain
    # http://theautomatic.net/2019/04/17/how-to-get-options-data-with-python/
    # '''

    # # option_chain(Symbols)
    # option_chain(Symbols)
    # option_chain_today(Symbols)



