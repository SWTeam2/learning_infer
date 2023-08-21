'''
이 코드는 사용 안해요. 혹시 몰라서 저장해놓음 
this code not using
'''

# import json
# import os
# import datetime
# import pickle as pkl
# import pywt
# import numpy as np
# import pandas as pd
# import requests
# import csv


# # get data from DB
# def get_df(url, table, id):
#     # request to DB
#     # REST API 경로에 접속하여 응답(Response) 데이터 받아오기
#     params = {'table': table, 'id': id}
#     url = url+f'{table}/{id}'
#     response = requests.get(url=url, params=params)
#     data = response.json()
#     # python_list = json.loads(data)
#     df = pd.DataFrame(data)
    
#     return df


# # perform CWT on 1d signals and return 2d feature image
# def extract_feature_image(df, feature_name='horiz_accel'):
#     # Parameters or Required Variables
#     DATA_POINTS_PER_FILE = 2560
#     WIN_SIZE = 20
#     WAVELET_TYPE = 'morl'
    
#     data = df[feature_name]
#     # use window to process(= prepare, develop) 1d signal
#     data = np.array([np.mean(data[i:i+WIN_SIZE])
#                     for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
#     # perform cwt on 1d data
#     coef, _ = pywt.cwt(data, np.linspace(1, 128, 128), WAVELET_TYPE)
#     # transform to power and apply logarithm ?!
#     coef = np.log2(coef**2+0.001)
#     # normalize coef
#     coef = (coef - coef.min())/(coef.max() - coef.min())
#     return coef


# # main util function
# def load_data(table, load_cnt):
#     '''
#     It receives vibration data through DB, converts it into an image, and returns it.
#     - load_cnt: The number of times the data was loaded, used for the name of the tmp .pkz
#     - host, database, user, password: Elements required for DB connection
#     '''
    
#     # id 계산
#     # this code is getting the 2560 row to make 1D to 2D image. 
#     DATA_POINTS_PER_FILE = 2560
#     id = (load_cnt-1)*DATA_POINTS_PER_FILE + 1
    
#     # make tmp pkz folder
#     if not os.path.exists('static'):
#         os.makedirs('static')
        
#     print(id)
    
#     # get data from DB
#     df = get_df('https://win1.i4624.tk/data/', table, id)
#     df = df.drop(columns=['id'])
#     print(df)
    
#     # signal processing = Extracting Time-Frequency Domain feature images
#     data = {'timestamps': [], 'x': []}
#     coef_h = extract_feature_image(df, feature_name='horiz_accel')
#     coef_v = extract_feature_image(df, feature_name='vert_accel')
#     x_ = np.array([coef_h, coef_v])
#     data['x'].append(x_)

#     # Create a datetime object with only time information
#     timestamp = datetime.datetime.min.time().replace(hour=df.loc[0, 'hour'], minute=df.loc[0, 'minutes'], second=df.loc[0, 'second']) # type: ignore
#     data['timestamps'].append(timestamp)
#     data['x'] = np.array(data['x']) # type: ignore
    
#     # load tmp data and append new data(as data is time series)
#     # Finally delete the read file
#     if load_cnt > 0:
#         pkz_file = os.path.join('static', f'{load_cnt-1}_tmp_bearing.pkz')
#         if os.path.exists(pkz_file):
#             with open(pkz_file, 'rb') as f:
#                 tmp_data = pkl.load(f)
        
#             # Determine how many data points to use from tmp_data
#             num_to_use = min(100, len(tmp_data['timestamps']))
#             tmp_data['timestamps'] = tmp_data['timestamps'][-num_to_use:]
#             tmp_data['x'] = tmp_data['x'][-num_to_use:]
            
#             data['timestamps'] = tmp_data['timestamps']
#             data['x'] = np.concatenate((data['x'], tmp_data['x'])) # type: ignore
#             # save tmp data
#             out_file = os.path.join('static', f'{load_cnt}_tmp_bearing.pkz')
#             with open(out_file, 'wb') as f:
#                 pkl.dump(data, f)
#         else:
#             pass  # Handle case where tmp_data file doesn't exist
#     # elif load_cnt > 0:
#     #     for i in range(1,5):
#     #         df = get_df('https://win1.i4624.tk/data/', table, DATA_POINTS_PER_FILE*i)
#     #         df = df.drop(columns=['id'])
#     #         # signal processing = Extracting Time-Frequency Domain feature images
#     #         data = {'timestamps': [], 'x': []}
#     #         coef_h = extract_feature_image(df, feature_name='horiz_accel')
#     #         coef_v = extract_feature_image(df, feature_name='vert_accel')
#     #         x_ = np.array([coef_h, coef_v])
#     #         data['x'].append(x_)

#     #         # Create a datetime object with only time information
#     #         timestamp = datetime.datetime.min.time().replace(hour=df.loc[0, 'hour'], minute=df.loc[0, 'minutes'], second=df.loc[0, 'second']) # type: ignore
#     #         data['timestamps'].append(timestamp)
#     #         data['x'] = np.array(data['x']) # type: ignore
#     #         out_file = os.path.join('static', f'{i}_tmp_bearing.pkz')
#     #         with open(out_file, 'wb') as f:
#     #             pkl.dump(data, f)
#     else:
#         pass

#     return data


# # df = load_data('test_table_bearing1_3', load_cnt=2)
# # print(np.shape(df['x']))