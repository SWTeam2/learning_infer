# file_utils.py

import os
import csv
import json
import tempfile
import datetime

def save_uploaded_file(file):
    """
    Save the uploaded file to a temporary directory.
    Args:
        file: The uploaded file.
    Returns:
        The path to the saved file.
    """
    filename = file.filename
    file_path = os.path.join(tempfile.gettempdir(), filename)
    with open(file_path, 'wb') as f:
        f.write(file.file.read())
    return file_path

def save_results(results, file_name, fig):
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # Save the results to a CSV file
    with open(os.path.join('results', file_name + time + '.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for key, value in results.items():
            writer.writerow([key, value])
    
    # Save the results to a JSON file
    with open(os.path.join('results', file_name + time + '.json'), 'w') as jsonfile:
        json.dump(results, jsonfile)

    jsonfile_path = os.path.join('results', file_name + time + '.json')

        # Save the plot image
    
    fig_bytes = fig.savefig(os.path.join('results/plots', time + '.png'), format='png')
    return jsonfile_path
