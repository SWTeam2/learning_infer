import uvicorn
from app import app_routing


## background running uvicorn --reload excute:app --port 48000 --host 0.0.0.0 
if __name__ == "__main__":
    uvicorn.run(app_routing.route, host="0.0.0.0", port=48000)
