'''
Prgram 명: act_best_port.py
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
최적의 포트폴리오 비중에 대한 검증 프로그램으로 현재의 경제상황(상승장, 하락장, 채권장, 현금확보장, 외화환전장...)에서 포트비중 산출 
- 주식, 채권, 원자재, 현금/예금 비중
History
20230106  Create
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

# logging
logger.warning(sys.argv[0] + ' :: ' + str(datetime.today()))
logger2.info('')
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))







'''
01. Simulate Multi-Asset Baskets With Correlated Price Paths
- https://medium.com/codex/simulate-multi-asset-baskets-with-correlated-price-paths-using-python-472cbec4e379
'''
def multiAsset_basket():
    sp500 = fred.get_series('SP500', observation_start=from_date_MT)    
    nasdaq_comp = fred.get_series('NASDAQCOM', observation_start=from_date_MT)
    oil_wti = fred.get_series('DCOILWTICO', observation_start=from_date_MT)
    bond_10y = fred.get_series('DGS10', observation_start=from_date_MT)
    dollar_idx = fred.get_series('DTWEXBGS', observation_start=from_date_MT)

    sp500_norm = (sp500-sp500.min()) / (sp500.max()-sp500.min())
    nasdaq_comp_norm = (nasdaq_comp-nasdaq_comp.min()) / (nasdaq_comp.max()-nasdaq_comp.min())
    oil_wti_norm = (oil_wti-oil_wti.min()) / (oil_wti.max()-oil_wti.min())
    bond_10y_norm = (bond_10y-bond_10y.min()) / (bond_10y.max()-bond_10y.min())
    dollar_idx_norm = (dollar_idx-dollar_idx.min()) / (dollar_idx.max()-dollar_idx.min())

    # Manually input number of stocks
    NUMBER_OF_ASSETS = 5
    ASSET_TICKERS = ["sp500", "Nasdaq", "Oil", "Bond_10y", "Dollar"]
    Vol_sp500 = sp500_norm.std()
    Vol_Nasdaq = nasdaq_comp_norm.std()
    Vol_oil = oil_wti_norm.std()
    Vol_bond_10y = bond_10y_norm.std()
    Vol_dollar = dollar_idx_norm.std()

    VOLATILITY_ARRAY =[Vol_sp500, Vol_Nasdaq, Vol_oil, Vol_bond_10y, Vol_dollar]
    temp = pd.DataFrame()
    temp= pd.concat([temp, sp500, nasdaq_comp, oil_wti, bond_10y, dollar_idx], axis=1)
    temp.columns = ['sp500_norm', 'nasdaq_comp_norm', 'oil_wti_norm', 'bond_10y_norm', 'dollar_idx_norm']
    temp.fillna(method='ffill', inplace=True)
    COEF_MATRIX = temp.corr()

    # Perform Cholesky decomposition on coefficient matrix
    R = np.linalg.cholesky(COEF_MATRIX)
    # Compute transpose conjugate (only for validation)
    RT = R.T.conj()
    # Reconstruct coefficient matrix from factorization (only for validation)
    logger2.info("Multi-Asset Baskets With Correlated Price".center(60, '*'))
    logger2.info(": \n" + str(COEF_MATRIX))
    # logger2.info(": \n" + str(np.dot(R, RT)))

    T = 250                                   # Number of simulated days
    asset_price_array = np.full((NUMBER_OF_ASSETS,T), 100.0) # Stock price, first value is simulation input 
    volatility_array = VOLATILITY_ARRAY       # Volatility (annual, 0.01=1%)
    r = 0.001                                 # Risk-free rate (annual, 0.01=1%)
    dt = 1.0 / T

    # Plot simulated price paths
    retry_cnt = 5
    fig = plt.figure(figsize=(16,4*retry_cnt))

    for i in range(retry_cnt):

        for t in range(1, T):
            # Generate array of random standard normal draws
            random_array = np.random.standard_normal(NUMBER_OF_ASSETS)
            # Multiply R (from factorization) with random_array to obtain correlated epsilons
            epsilon_array = np.inner(random_array,R)
            # Sample price path per stock
            for n in range(NUMBER_OF_ASSETS):
                dt = 1 / T 
                S = asset_price_array[n,t-1]
                v = volatility_array[n]
                epsilon = epsilon_array[n]
                # Generate new stock price
                if n == 0:
                    asset_price_array[n,t] = S * exp((r - 0.5 * v**2) * dt + v * sqrt(dt) * epsilon)
                else:
                    asset_price_array[n,t] = temp.iloc[t,n]+100*1
                asset_price_array[n,t] = S * exp((r - 0.5 * v**2) * dt + v * sqrt(dt) * epsilon)

        ax = fig.add_subplot(retry_cnt, 1,  i+1)
        array_day_plot = [t for t in range(T)]
        for n in range(NUMBER_OF_ASSETS):
            ax.plot(array_day_plot, asset_price_array[n],\
                                label = '{}'.format(ASSET_TICKERS[n]))

        plt.grid()
        plt.xlabel('Day')
        plt.ylabel('Asset price')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(reports_dir + '/best_port_0100.png')




'''
Main Fuction
'''

if __name__ == "__main__":

    '''
    0. 공통
    '''
    multiAsset_basket()
