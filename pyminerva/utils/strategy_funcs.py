# Copyright 2023-2025 Jeongmin Kang, jarvisNim @ GitHub
# See LICENSE for details.


from datetime import datetime, timedelta
from pytz import timezone

from . import constant as cst


def find_30days_ago():
    # 'America/New_York' 타임존으로 설정
    ny_tz = timezone('America/New_York')
    # 현재 시간을 얻어옴
    now = datetime.now(ny_tz)    
    # 30일 전의 날짜를 계산
    day30_ago = now - timedelta(days=30)
    # day30_ago를 'UTC' 타임존으로 변환
    day30_ago_utc = day30_ago.astimezone(timezone('UTC'))
    
    return day30_ago_utc