import datetime
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from app.cnn_lstm import CNN_LSTM_FP, infer_model, series_infer
from app.data import load_data_from_pfile, PHMTestDataset_Sequential
from app.file_utils import save_results, save_uploaded_file
from app.db_connection import connect_db, insert_data, disconnect_db 
from app.test_dataset_preparation import load_data
import torch
import matplotlib.pyplot as plt
from pydantic import BaseModel

import json

route = FastAPI()

def dbworks(csvfile_path, current_time):
    conn = connect_db()
    insert_data(conn, csvfile_path, current_time)

@route.post("/local_predict/")
async def predict(file: UploadFile):
    # Read the file contents
    file_path = save_uploaded_file(file)

    # Load the data from the file
    sample_data = load_data_from_pfile(file_path)

    # Load the PyTorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_FP().to(device)
    model.load_state_dict(torch.load('../model/weight/bearing(1,noisy1) + bearing(2,noisy2)(0.25,0.3)_model.pth', map_location=device))

    # Do the inference
    results = infer_model(model, file_path, device)
    results['timestamps'] = sample_data['timestamps']

    results['timestamps'] = [ts.strftime('%H:%M:%S') for ts in results['timestamps']]

    current_time = datetime.datetime.now().replace(microsecond=0)
    # Plotting and saving
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
    ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
    ax.legend()
    filename = 'inf_result'
    jsonfile_path, csvfile_path = save_results(results, filename, fig,current_time)
    # Load the prediction results from the original JSON file

    dbworks(csvfile_path, current_time)
    
    return FileResponse(
        path=jsonfile_path,
        media_type='application/json'
    )



@route.get("/predict/")
async def seriesPredict(table: str, load_cnt: int):
    sample_data = load_data(table, load_cnt)

    # Load the PyTorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_FP().to(device)
    model.load_state_dict(torch.load('../model/weight/bearing(1,noisy1) + bearing(2,noisy2)(0.25,0.3)_model.pth', map_location=device))

    # Do the inference
    results = series_infer(model, load_cnt, device)
    results['timestamps'] = sample_data['timestamps']

    results['timestamps'] = [ts.strftime('%H:%M:%S') for ts in results['timestamps']]

    current_time = datetime.datetime.now().replace(microsecond=0)
    # Plotting and saving
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
    ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
    ax.legend()
    filename = 'inf_result'
    
    jsonfile_path, csvfile_path = save_results(results, filename, fig, current_time) ## folder input fix required 
    # Load the prediction results from the original JSON file


    
    return FileResponse(
        path=jsonfile_path,
        media_type='application/json'
    )