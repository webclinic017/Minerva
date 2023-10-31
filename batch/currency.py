'''
Prgram 명: Cracks for Currency
Author: jimmy Kang
Mail: jarvisNim@gmail.com
외화환율부문에서 신뢰구간을 벗어나는지 Daily Check 하는 목적

History
2022/08/19  Create
2022/08/20  today record delete 함수 추가
2022/08/23  외환지수 그래프 보완 (달러/원화/위안/엔)
'''

# common area import
import sys
sys.path.append('../Utils')
from settings import *

# local import
import re
import seaborn as sns
import plotly.express as px
import requests
from bs4 import BeautifulSoup as bs
import html5lib

# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0])

###################################################################################################
# 모니터링 테이블 (Sent_Crack) 생성
###################################################################################################
# Connect DataBase
database = database_dir+'/Crack_Curr.db'
engine = 'sqlite:///' + database
conn = create_connection(database)

# 환율 테이블 생성
def create_Curr_Crack(conn):
    with conn:
        cur=conn.cursor()
        cur.execute('create table if not exists Curr_Crack (Date text primary key, Tot_Percent real, \
            Tot_Count integer, CRCUR0001 integer, CRCUR0002 integer, CRCUR0003 integer, CRCUR0004 integer, \
            CRCUR0005 integer, CRCUR0006 integer, CRCUR0007 integer, CRCUR0008 integer, CRCUR0009 integer, \
            CRCUR0010 integer)')

    return conn

create_Curr_Crack(conn)
today = now.strftime('%Y-%m-%d')
M_table = 'Curr_Crack'
M_query = f"SELECT * from {M_table}"
try:
    # 오늘자 Dataframe, db는 테이블에 있는 Dataframe 읽어온거.
    M_db = pd.read_sql_query(M_query, conn)
    buf = [today, 0,0,0,0,0,0,0,0,0,0,0,0]
    _M_db = pd.DataFrame(data=[buf], columns=M_db.columns)
    logger2.info(M_db[-5:])
except Exception as e:
    print('Exception: {}'.format(e))


'''

'''

