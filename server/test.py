import datetime
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from app.cnn_lstm import CNN_LSTM_FP, infer_model, series_infer
from app.data import load_data_from_pfile, PHMTestDataset_Sequential
from app.file_utils import save_results, save_uploaded_file

from app.test_dataset_preparation import load_data
import torch
import matplotlib.pyplot as plt
from pydantic import BaseModel

import json

def seriesPredict(table: str, load_cnt: int):
    sample_data = load_data(table, load_cnt)
    
    # Load the PyTorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_FP().to(device)
    model.load_state_dict(torch.load('/Users/jeongho/github/learning_infer/model/weight/bearing(1,noisy1) + bearing(2,noisy2)(0.25,0.3)_model.pth', map_location=device))

    # Do the inference
    results = series_infer(model, device, table, load_cnt)
    results['timestamps'] = sample_data['timestamps']
    results['timestamps'] = [ts.strftime('%H:%M:%S') for ts in results['timestamps']]
    return results

    # current_time = datetime.datetime.now().replace(microsecond=0)
    # # Plotting and saving
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
    # ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions') # type: ignore
    # ax.legend()
    # filename = 'inf_result'
    
    # jsonfile_path, csvfile_path = save_results(results, filename, fig, current_time) ## folder input fix required 
    # # Load the prediction results from the original JSON file

    
    # return FileResponse(
    #     path=jsonfile_path,
    #     media_type='application/json'
    # )
import numpy as np
a = seriesPredict('test_table_bearing1_3', 1)
print(a)
