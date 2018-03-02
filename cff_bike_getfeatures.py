# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:16:40 2017
注意：LEASEDTIME  是借车时间
@author: tangwenhua
"""

import pickle
import os
import pandas as pd
import time,datetime
import math


#数据文件
Train_INDEX = "D:/J_data/CFF_Bike/index_all.csv"
All_DATA = "D:/J_data/CFF_Bike/train.csv"
Test_DATA = "D:/J_data/CFF_Bike/index.csv"
SHED_CLASS =  "D:/J_data/CFF_Bike/train_class.csv"
WEATHER =  "D:/J_data/CFF_Bike/pro_weather_yancheng.csv"

def get_index_train(start_date,end_date):
    
    index = pd.read_csv(Train_INDEX, header=0, encoding="gbk")
    
    index = index[(index.time >= start_date) & (index.time <= end_date)]
    
    index.sort_values(["time"],ascending=True)
    index = index.reset_index(drop=True)
    return index

#获得训练集的索引，即每个样本的索引
def get_index_test():
    
    index = pd.read_csv(Test_DATA, header=0, encoding="gbk")
    
    return index
     
def get_date_feature(start_date,end_date):
    
    dump_path = 'D:/J_data/CFF_Bike/cache/basic_user.pkl'
    if os.path.exists(dump_path):
        index = pickle.load(open(dump_path,'rb'))
    else:
        index = get_index_train(start_date,end_date);
        #判断该天是星期几并one-hot编码
        index['dayOfWeek_df'] = index['time'].map(convert_str_to_dayofweek)
        dayOfWeek_df = pd.get_dummies(index["dayOfWeek_df"], prefix="dayOfWeek")
        index = pd.concat([index, dayOfWeek_df], axis=1)
        
        index['isFestival'] = index['time'].map(isFestival).fillna(0)  
        del index['dayOfWeek_df']

        #判断它是上午还是下午
        index['amorpm'] = index['time'].map(am_or_pm)
        
        #判断到底是哪个站点
        index['idshed'] = index['SHEDID']
        
        #判断该天是该月的第几天
        index['dayOfmonth'] = index['time'].map(dayofmonth)
        index = index.reset_index(drop=True)
          
    return index

def get_all_feature(start_date,end_date):
    data_feature = get_date_feature(start_date,end_date)
    weather_feature = get_weather_feature()
    shed_feature_1 = get_shed_Lease_feature()
    shed_feature_2 = get_shed_RT_feature()
    all_feature = pd.merge(data_feature, weather_feature, how='left', on=['time']).fillna(0)
    all_feature = pd.merge(all_feature, shed_feature_1, how='left', on=['SHEDID']).fillna(0)
    all_feature = pd.merge(all_feature, shed_feature_2, how='left', on=['SHEDID']).fillna(0)
    
    return all_feature
        
    
def get_weather_feature():
    dump_path = 'D:/J_data/CFF_Bike/cache/basic_user.pkl'
    if os.path.exists(dump_path):
        index = pickle.load(open(dump_path,'rb'))
    else:
        index = pd.read_csv(WEATHER, header=0, encoding="gbk")
        del index['wind_direction_0']
        del index['wind_direction_1']
        del index['wind_direction_2']
        del index['wind_direction_3']
        del index['wind_direction_4']
        del index['wind_direction_5']
        del index['wind_direction_6']
    return index
    
    
def get_shed_Lease_feature():
    dump_path = 'D:/J_data/CFF_Bike/cache/basic_user.pkl'
    if os.path.exists(dump_path):
        index = pickle.load(open(dump_path,'rb'))
    else:
        index = pd.read_csv(SHED_CLASS, header=0, encoding="gbk")
        index = pd.concat([index['SHEDID'], index['LEASE_CLASS']], axis=1)
        index = index.drop_duplicates(['SHEDID'])
        index = index.reset_index(drop=True)
    return index

def get_shed_RT_feature():
    dump_path = 'D:/J_data/CFF_Bike/cache/basic_user.pkl'
    if os.path.exists(dump_path):
        index = pickle.load(open(dump_path,'rb'))
    else:
        index = pd.read_csv(SHED_CLASS, header=0, encoding="gbk")
        index = pd.concat([index['SHEDID'], index['RT_CLASS']], axis=1)
        index = index.drop_duplicates(['SHEDID'])
        index = index.reset_index(drop=True)
    return index

def convert_str_to_dayofweek(mytimes):
    
    t = time.strptime(mytimes, "%Y-%m-%d-%H")
    y,m,d = t[0:3]
    t = datetime.datetime(y,m,d).weekday()
    
    return t
    
def isFestival(mytimes):
    t = time.strptime(mytimes, "%Y-%m-%d-%H")
    mon = t.tm_mon
    day = t.tm_mday
    if (mon == 5) & (day == 1):
        return 1;
    if (mon == 5) & (day == 2):
        return 1;
    if (mon == 5) & (day == 3):
        return 1;
    if (mon == 6) & (day == 20):
        return 1;
    if (mon == 6) & (day == 21):
        return 1;
    if (mon == 6) & (day == 22):
        return 1;
    if (mon == 8) & (day == 20):
        return 1;

#打标签(还车)
def get_lables_RT(start_date, end_date):

    all_data = pd.read_csv(All_DATA, header=0, encoding="gbk")
    all_data['mytime'] = all_data['LEASEDATE'].map(convert_str_to_data)+'-'+all_data['RTTIME'].map(convert_str_to_time)
    all_data = all_data[(all_data.mytime >= start_date) & (all_data.mytime <= end_date)]
    label_rt = all_data.groupby(['SHEDID','mytime']).size().reset_index()
    label_rt.rename(columns={'mytime':'time'}, inplace = True)
    return label_rt

    
#打标签(借车) 
def get_lables_LEASE(start_date, end_date):
    
    all_data = pd.read_csv(All_DATA, header=0, encoding="gbk")
    all_data['mytime'] = all_data['RTDATE'].map(convert_str_to_data)+'-'+all_data['LEASETIME'].map(convert_str_to_time)
    all_data = all_data[(all_data.mytime >= start_date) & (all_data.mytime <= end_date)]
    label_lease = all_data.groupby(['SHEDID','mytime']).size().reset_index()
    label_lease.rename(columns={'mytime':'time'}, inplace = True)
    label_lease = label_lease.reset_index(drop=True)
    return label_lease
    
    
def convert_str_to_mytime(mytimes):
    t = time.strptime(mytimes, "%m/%d/%Y")
    index = time.strftime("%Y-%m-%d", t)
    return index  
    
def convert_str_to_APtime0(mytimes):
    t = time.strptime(mytimes, "%m/%d/%Y")
    index = time.strftime("%Y-%m-%d", t)
    return index + "-0"  

def convert_str_to_APtime1(mytimes):
    t = time.strptime(mytimes, "%m/%d/%Y")
    index = time.strftime("%Y-%m-%d", t)
    return index + "-1" 

def convert_str_to_data(mytimes):
    
    t = time.strptime(mytimes, "%m/%d/%Y")
    index = time.strftime("%Y-%m-%d", t)
    
    return index
    
def convert_str_to_time(mytimes):
    
    t = time.strptime(mytimes, "%H:%M:%S")
    if t.tm_hour > 12:
        return "1"
    else:
        return "0"
        
def convert_time(mystr):
    tmp=list(mystr)
    tmp[6]='5'
    s=''.join(tmp)
    return s        

def am_or_pm(mystr):
    if mystr[-1] == '1':
        return 1
    if mystr[-1] == '0':
        return 0

def dayofmonth(mystr):
    t = time.strptime(mystr, "%Y-%m-%d-%H")
    return t.tm_mday


#俩个参数都是pandas的一列
def evaluate(ture_value, model_value):
    score = (((ture_value - model_value) * (ture_value - model_value)).sum()) / len(ture_value)
    score = math.sqrt(score)
    score = 1 / (1 + score)
    return score
