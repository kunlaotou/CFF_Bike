# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:52:10 2017

@author: tangwenhua
"""

import pandas as pd
from cff_bike_getfeatures import get_lables_RT
from cff_bike_getfeatures import get_lables_LEASE
from cff_bike_getfeatures import get_all_feature



def gen_train_feature_label():
    
    train_start_date = '2015-05-01'
    train_end_date = '2015-07-15'
    test_start_date = '2015-07-16'
    test_end_date = '2015-07-31'
    
    train_X = get_all_feature(train_start_date,train_end_date)
    train_lease_label_y = get_lables_LEASE(train_start_date,train_end_date)
    train_rt_label_y = get_lables_RT(train_start_date,train_end_date)
    
    test_X = get_all_feature(test_start_date,test_end_date)
    test_lease_label_Y = get_lables_LEASE(test_start_date,test_end_date)
    test_rt_label_Y = get_lables_RT(test_start_date,test_end_date)
    
    train_lease_label = pd.merge(train_X, train_lease_label_y, how='left', on=['SHEDID', 'time']).fillna(0)
    train_rt_label = pd.merge(train_X, train_rt_label_y, how='left', on=['SHEDID', 'time']).fillna(0)
    
    test_lease_label = pd.merge(test_X, test_lease_label_Y, how='left', on=['SHEDID', 'time']).fillna(0)
    test_rt_label = pd.merge(test_X, test_rt_label_Y, how='left', on=['SHEDID', 'time']).fillna(0)
    
    return train_lease_label,train_rt_label,test_lease_label,test_rt_label


    
train_lease_label, train_rt_label, test_lease_label, test_rt_label = gen_train_feature_label()



train_lease_label.to_csv('D:/J_data/CFF_Bike/cache/train_lease_label.csv', index=False, index_label=False)
train_rt_label.to_csv('D:/J_data/CFF_Bike/cache/train_rt_label.csv', index=False, index_label=False)
test_lease_label.to_csv('D:/J_data/CFF_Bike/cache/test_lease_label.csv', index=False, index_label=False)
test_rt_label.to_csv('D:/J_data/CFF_Bike/cache/test_rt_label.csv', index=False, index_label=False)


#8月份特征
feature_08 = get_all_feature('2015-08-01','2015-08-31')
feature_08.to_csv('D:/J_data/CFF_Bike/cache/feature_08.csv', index=False, index_label=False)
