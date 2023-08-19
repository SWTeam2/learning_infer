import json
import os
import datetime
import pickle as pkl
import pywt
import numpy as np
import pandas as pd
import requests
import csv


# get data from DB
def get_df(url, table, id):
    # request to DB
    # REST API 경로에 접속하여 응답(Response) 데이터 받아오기
    params = {'table': table, 'id': id}
    url = url+f'{table}/{id}'
    response = requests.get(url=url, params=params)
    data = response.json()
    # python_list = json.loads(data)
    df = pd.DataFrame(data)
    
    return df


# perform CWT on 1d signals and return 2d feature image
def extract_feature_image(df, feature_name='horiz_accel'):
    # Parameters or Required Variables
    DATA_POINTS_PER_FILE = 2560
    WIN_SIZE = 20
    WAVELET_TYPE = 'morl'
    
    data = df[feature_name]
    # use window to process(= prepare, develop) 1d signal
    data = np.array([np.mean(data[i:i+WIN_SIZE])
                    for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
    # perform cwt on 1d data
    coef, _ = pywt.cwt(data, np.linspace(1, 128, 128), WAVELET_TYPE)
    # transform to power and apply logarithm ?!
    coef = np.log2(coef**2+0.001)
    # normalize coef
    coef = (coef - coef.min())/(coef.max() - coef.min())
    return coef


# main util function
def load_data(table, load_cnt=1):
    '''
    It receives vibration data through DB, converts it into an image, and returns it.
    - load_cnt: The number of times the data was loaded, used for the name of the tmp .pkz
    - host, database, user, password: Elements required for DB connection
    '''
    
    # id 계산
    DATA_POINTS_PER_FILE = 2560
    id = (load_cnt-1)*DATA_POINTS_PER_FILE + 1
    
    # get data from DB
    df = get_df('https://win1.i4624.tk/data/', table, id)

    # signal processing = Extracting Time-Frequency Domain feature images
    data = {'timestamps': [], 'x': []}
    coef_h = extract_feature_image(df, feature_name='horiz_accel')
    coef_v = extract_feature_image(df, feature_name='vert_accel')
    x_ = np.array([coef_h, coef_v])
    data['x'].append(x_)

    # Create a datetime object with only time information
    timestamp = datetime.datetime.min.time().replace(hour=df.loc[0, 'hour'], minute=df.loc[0, 'minutes'], second=df.loc[0, 'second'])
    data['timestamps'].append(timestamp)
    data['x'] = np.array(data['x'])
    
    # load tmp data and append new data(as data is time series)
    # Finally delete the read file
    if load_cnt > 1:
        pkz_file = os.path.join('static', f'{load_cnt-1}_tmp_bearing.pkz')
        with open(pkz_file, 'rb') as f:
            tmp_data = pkl.load(f)
        tmp_data['timestamps'].append(timestamp)
        data['timestamps'] = tmp_data['timestamps']
        data['x'] = np.concatenate((data['x'], tmp_data['x']))
    else:
        pass

    # save tmp data
    out_file = os.path.join('static', f'{load_cnt}_tmp_bearing.pkz')
    with open(out_file, 'wb') as f:
        pkl.dump(data, f)
    return data


# df = load_data('test_table_bearing1_3', load_cnt=2)
# print(np.shape(df['x']))