import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import urllib.request
import datetime
import json
import glob
import sys
import os

from prophet import Prophet

import warnings
warnings.filterwarnings(action='ignore')

# %matplotlib inline
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'Malgun Gothic' 20231006
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.grid'] = False

pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 250)
pd.set_option('display.width', 100)

pd.options.display.float_format = '{:.2f}'.format


class NaverDLabApi():
    """
    네이버 데이터랩 컨트롤러
    """
    
    def __init__(self, client_id, client_secret):
        """
        인증기 설정 및 검색어 그룹 초기화
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.keywordGroups = []
        self.url = "https://openapi.naver.com/v1/datalab/search"
        
        
    def add_keyword_groups(self, group_dict):
        """
        검색어 그룹 추가
        """
        
        keyword_group = {
            'groupName': group_dict['groupName'],
            'keywords': group_dict['keywords']
        }
        
        self.keywordGroups.append(keyword_group)
        # print(f">>> Num of keywordGroups: {len(self.keywordGroups)}")
        
             
    def get_data(self, from_date, to_date, time_unit, device, ages, gender):
        # Request Body
        body = json.dumps({
            'startDate': from_date,
            'endDate': to_date,
            'timeUnit': time_unit,
            'keywordGroups': self.keywordGroups,
            'device': device,
            'ages': ages,
            'gender': gender
        }, ensure_ascii=False)
        
        # Response
        request = urllib.request.Request(self.url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)
        request.add_header("Content-Type","application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))

        rescode = response.getcode()
        if(rescode==200):
            result = json.loads(response.read())
            df = pd.DataFrame(result['results'][0]['data'])[['period']]
            for i in range(len(self.keywordGroups)):
                tmp = pd.DataFrame(result['results'][i]['data'])
                tmp = tmp.rename(columns={'ratio': result['results'][i]['title']})
                df = pd.merge(df, tmp, how='left', on=['period'])
            self.df = df.rename(columns={'period': 'Date'})
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        else:
          print("Error Code:" + rescode)
        
        return self.df


    def plot_daily_trend(self):
        """
        일 별 검색어 트렌드 그래프 출력
        """
        colList = self.df.columns[1:]
        n_col = len(colList)

        fig = plt.figure(figsize=(12,6))
        plt.title('Daily Trend', size=20, weight='bold')
        for i in range(n_col):
            sns.lineplot(x=self.df['Date'], y=self.df[colList[i]], label=colList[i])
        plt.legend(loc='upper right')
        plt.grid()
        
        return fig
    

    def plot_monthly_trend(self):
        """
        월 별 검색어 트렌드 그래프 출력
        """
        df = self.df.copy()
        df_0 = df.groupby(by=[df['Date'].dt.year, df['Date'].dt.month]).mean().droplevel(0).reset_index().rename(columns={'Date': 'Month'})
        df_1 = df.groupby(by=[df['Date'].dt.year, df['Date'].dt.month]).mean().droplevel(1).reset_index().rename(columns={'Date': 'Year'})

        df = pd.merge(df_1[['Year']], df_0, how='left', left_index=True, right_index=True)
        df['Date'] = pd.to_datetime(df[['Year','Month']].assign(Day=1).rename(columns={"Year": "year", "Month":'month','Day':'day'}))
        
        colList = df.columns.drop(['Date','Year','Month'])
        n_col = len(colList)

        fig = plt.figure(figsize=(12,6))
        plt.title('Monthly Trend', size=20, weight='bold')
        for i in range(n_col):
            sns.lineplot(x=df['Date'], y=df[colList[i]], label=colList[i])
        plt.legend(loc='upper right')
        plt.grid()
        
        return fig


    def plot_pred_trend(self, days):
        """
        검색어 그룹 별 시계열 트렌드 예측 그래프 출력
        days: 예측일수
        """
        colList = self.df.columns[1:]
        n_col = len(colList)
        
        fig_list = []
        for i in range(n_col):
            
            globals()[f"df_{str(i)}"] = self.df[['Date', f'{colList[i]}']]
            globals()[f"df_{str(i)}"] = globals()[f"df_{str(i)}"].rename(columns={'Date': 'ds', f'{colList[i]}': 'y'})

            m = Prophet()
            m.fit(globals()[f"df_{str(i)}"])

            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            
            globals()[f"fig_{str(i)}"] = m.plot(forecast, figsize=(12,6))
            plt.title(colList[i], size=20, weight='bold')
            
            fig_list.append(globals()[f"fig_{str(i)}"])
            
        return fig_list


    # def value_to_float(x):
    #     if type(x) == float or type(x) == int:
    #         return x
    #     elif 'K' in x:
    #         if len(x) > 1:
    #             return float(x.replace('K', '')) * 1000
    #         return 1000.0
    #     elif 'M' in x:
    #         if len(x) > 1:
    #             return float(x.replace('M', '')) * 1000000
    #         return 1000000.0
    #     elif 'B' in x:
    #         return float(x.replace('B', '')) * 1000000000
    #     else:
    #         return float(x)