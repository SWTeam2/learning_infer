from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from app.model import CNN_LSTM_FP, infer_model
from app.data import load_data_from_pfile, PHMTestDataset_Sequential
from app.file_utils import save_results, save_uploaded_file
import torch
import matplotlib.pyplot as plt

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    # Read the file contents
    file_path = save_uploaded_file(file)

    # Load the data from the file
    sample_data = load_data_from_pfile(file_path)

    # Load the PyTorch model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_FP().to(device)
    model.load_state_dict(torch.load('../model/weight/cnn_lstm_model_gpu2_epoch50_batch32.pth', map_location=device))

    # Do the inference
    results = infer_model(model, file_path, device)
    results['timestamps'] = sample_data['timestamps']

    results['timestamps'] = [ts.strftime('%H:%M:%S') for ts in results['timestamps']]

    # Plotting and saving
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 10])
    ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
    ax.legend()
    filename = 'inf_result'
    jsonfile = save_results(results, filename, fig)

    return FileResponse(
        path=jsonfile,
        media_type='application/json'
    )

# Rest of the code remains the same
