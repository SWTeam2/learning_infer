import os
import datetime
import pickle as pkl
import pywt
import numpy as np
import pandas as pd


# get data from DB
def get_df(host, database, user, password, DATA_POINTS_PER_FILE):
    # request to DB
    
    
    # get new csv from DB
    colname = ['hour', 'minute', 'second', 'microsecond', 'horiz accel', 'vert accel']
    # df = 
    
    
    return df

# perform CWT on 1d signals and return 2d feature image
def extract_feature_image(ind, feature_name='horiz accel'):
    data_range = (DATA_POINTS_PER_FILE*ind, DATA_POINTS_PER_FILE*(ind+1))
    data = df[feature_name].values[data_range[0]:data_range[1]]
    # use window to process(= prepare, develop) 1d signal
    data = np.array([np.mean(data[i:i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])  
    # perform cwt on 1d data
    coef, _ = pywt.cwt(data, np.linspace(1,128,128), WAVELET_TYPE)  
    # transform to power and apply logarithm ?!
    coef = np.log2(coef**2+0.001) 
    # normalize coef
    coef = (coef - coef.min())/(coef.max() - coef.min()) 
    return coef

def load_pkz(pkz_file):
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

##### main util function
def load_data(host, database, user, password, load_cnt=0):
    '''
    It receives vibration data through DB, converts it into an image, and returns it.
    - load_cnt: The number of times the data was loaded, used for the name of the tmp .pkz
    - host, database, user, password: Elements required for DB connection
    '''
    # Parameters or Required Variables
    DATA_POINTS_PER_FILE = 2560
    TIME_PER_REC = 0.1
    SAMPLING_FREQ = 25600 # 25.6 KHz
    SAMPLING_PERIOD = 1.0/SAMPLING_FREQ

    WIN_SIZE = 20
    WAVELET_TYPE = 'morl'

    # file path
    main_dir = 'server'
    
    # get data from DB
    ###### DB에서 받는 걸로 함수 수정하기
    df=get_df(host, database, user, password, DATA_POINTS_PER_FILE)

    # signal processing = Extracting Time-Frequency Domain feature images
    data = {'timestamps': [], 'x': []}
    for i in range(0, no_of_files):
        coef_h = extract_feature_image(i, feature_name='horiz accel')
        coef_v = extract_feature_image(i, feature_name='vert accel')
        x_ = np.array([coef_h, coef_v])
        data['x'].append(x_)
        
        # Create a datetime object with only time information
        idx = i*DATA_POINTS_PER_FILE
        timestamp = datetime.datetime.min.time().replace(hour=df.iloc[idx,0], minute=df.iloc[idx,1], second=df.iloc[idx,2])
        data['timestamps'].append(timestamp)
        
    data['x']=np.array(data['x'])
    assert data['x'].shape==(no_of_files, 2, 128, 128)
    
    # load tmp data and append new data(as data is time series)
    if load_cnt > 0:
        pkz_file = os.path.join(main_dir, 'static', f'{load_cnt-1}tmp_bearing1_4_test_data.pkz')
        tmp_data = load_pkz(pkz_file)
        data['timestamps'] = tmp_data['timestamps'].extend(data['timestamps'])
        data['x'] = tmp_data['x'].extend(data['x'])
    else:
        pass
    
    # save tmp data
    out_file = os.path.join(main_dir, 'static', f'{load_cnt}tmp_bearing1_4_test_data.pkz')
    with open(out_file, 'wb') as f:
        pkl.dump(data, f)
    
    return data