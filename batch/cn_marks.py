'''
Prgram 명: 중국, 홍콩 시장부문의 투자 전략
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
마켓은 주식/채권/원자재/현금등의 금융자산의 오픈된 시장을 의미하며, 
이 시장내에서의 다양한 개별적 전략(Strategy)들을 수립하고 이에 대한 백테스트 결과까지 도출하고 
주기적으로 이를 검증하며 매수/매도 기회를 포착하는 것을 목적으로 함. 
History
2023/12/12  Creat
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
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

'''
1. hanseng H index PBR 0.8 이하 매수, 1.2 매도 전략
'''
