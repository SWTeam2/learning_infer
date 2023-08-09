import datetime
import io
import fastapi
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import multipart
import torch
import requests
import uvicorn
import tempfile
import csv
import json

import base64
import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, RBF, RationalQuadratic


app = FastAPI()

# timestamp열을 추가해서 csv, json 파일로 결과와 이미지를 저장
def save_results(results, file_name, fig):
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    with open(os.path.join('results', file_name + time + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key, value in results.items():
            writer.writerow([key, value])

    with open(os.path.join('results', file_name + time + '.json'), 'w') as jsonfile:
        json.dump(results, jsonfile)

    jsonfile_path = os.path.join('results', file_name + time + '.json')
    
    fig_bytes = fig.savefig(os.path.join('results/plots', time + '.png'), format='png')
    return jsonfile_path

@app.post("/predict")
# 업로드된 파일을 미리 학습된 모델을 사용하여 추론후 그 결과물(csv,json)파일을 반환
async def predict(file: UploadFile):

    file_path = save_uploaded_file(file)

    sample_data = load_data_from_pfile(file_path)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    model = CNN_LSTM_FP().to(device)
    model.load_state_dict(torch.load('../model/weight/cnn_lstm_model_gpu2_epoch50_batch32.pth',map_location=device))

    results = infer_model(model, file_path, device)
    results['timestamps'] = sample_data['timestamps']

    results['timestamps'] = [ts.strftime('%H:%M:%S') for ts in results['timestamps']]
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,10])
    ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
    ax.legend()
    filename = 'inf_result'

    jsonfile = save_results(results , filename , fig)
    
    return FileResponse(
        path=jsonfile,
        media_type='application/json'
    )

# 업로드된 파일을 임시 디렉토리에 저장
def save_uploaded_file(file):

    filename = file.filename
    file_path = os.path.join(tempfile.gettempdir(), filename)
    with open(file_path, 'wb') as f:
        f.write(file.file.read())
    return file_path

# 업로드한 파일을 불러옴
def load_data_from_pfile(file_path):  

    with open(file_path, 'rb') as pfile:
        sample_data = pkl.load(pfile)
    return sample_data




# PHM 데이터를 다루기 위한 시퀀스 형태의 데이터셋 클래스 정의
class PHMTestDataset_Sequential(Dataset):
    def __init__(self, dataset='', seq_len=5):
        self.data = load_data_from_pfile(dataset)
        self.seq_len = seq_len
    
    def __len__(self):
        return self.data['x'].shape[0]-self.seq_len+1
    
    def __getitem__(self, i):
        sample = {'x': torch.from_numpy(self.data['x'][i:i+self.seq_len])}
        return sample

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', batch_norm=True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    nn.init.xavier_uniform_(conv.weight)
    relu = nn.ReLU()
    if batch_norm:
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            relu
        )
    else:
        return nn.Sequential(
            conv,
            relu
        )

class CNN_CWT_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_bn_relu(2, 16, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = conv_bn_relu(16, 32, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = conv_bn_relu(32, 64, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = conv_bn_relu(64, 128, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.pool1(x) 
        x = self.conv2(x) 
        x = self.pool2(x) 
        x = self.conv3(x) 
        x = self.pool3(x) 
        x = self.conv4(x)
        x = self.pool4(x) 
        x = self.flatten(x) 
        x = self.fc1(x) 
       
        x = nn.ReLU()(x)
        x = self.fc2(x) 
        x = nn.ReLU()(x) 
        return x


class CNN_LSTM_FP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNN_CWT_Encoder()
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        x = x.view(batch_size*seq_len, C, H, W) 
        x = self.encoder(x)
        x = x.view(batch_size, seq_len, -1) 
        x, _ = self.lstm1(x)
        x = self.fc(x[:,-1,:]) 
        x = nn.Sigmoid()(x)

        return x    



def model_inference_helper(model, dataloader, device):
    results = {'predictions':[]}
    model.eval()
    for i, batch in enumerate(dataloader):
        x = batch['x'].to(device, dtype=torch.float)

        with torch.no_grad():
            y_prediction = model(x)

        if y_prediction.size(0)>1:
            results['predictions'] += y_prediction.cpu().squeeze().tolist()
        elif y_prediction.size(0)==1:
            results['predictions'].append(y_prediction.cpu().squeeze().tolist())
    return results

def infer_model(model, file, device):
    test_dataset = PHMTestDataset_Sequential(dataset=file)

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    results = model_inference_helper(model, test_dataloader, device)

  

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)