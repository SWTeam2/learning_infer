import uvicorn
from app import app_routing



if __name__ == "__main__":
    uvicorn.run(app_routing.route, host="0.0.0.0", port=48000)
