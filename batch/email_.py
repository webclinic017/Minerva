'''
Prgram 명: email.py
Author: jeongmin Kang
Mail: jarvisNim@gmail.com
배치작업 결과를 e-mail 로 전송하는 프로그램
History
20240102  Create
'''

import sys, os
utils_dir = os.getcwd() + '/batch/utils'
sys.path.append(utils_dir)

from settings import *

'''
0. 공통영역 설정
'''
import smtplib
from email.mime.text import MIMEText


# logging
logger.warning(sys.argv[0])
logger2.info(sys.argv[0] + ' :: ' + str(datetime.today()))

# 3개월단위로 순차적으로 읽어오는 경우의 시작/종료 일자 셋팅
to_date_2 = pd.to_datetime(to_date2)
three_month_days = relativedelta(weeks=12)
from_date = (to_date_2 - three_month_days).date()
to_date_2 = to_date_2.date()


file_path = 'batch/reports/reports.log'

from_address = "master@meraelabs.com"
to_address = "jarvisnim@gmail.com"

smtp = smtplib.SMTP('smtp.gmail.com', 587)

smtp.ehlo()
smtp.starttls()
smtp.login('jarvisnim@gmail.com', 'uubn dcfc tevw dezs')

# 파일에서 텍스트 읽기
with open(file_path, 'r', encoding='UTF-8') as file:
    text_content = file.read()

msg = MIMEText(text_content)
msg['Subject'] = 'Pyminerva Batch Reports:'

smtp.sendmail(from_address, to_address, msg.as_string())

smtp.quit()