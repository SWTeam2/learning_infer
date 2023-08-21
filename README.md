# learning_infer

RUL 예측을 위한 모델을 학습하고 이를 추론해서 데이터베이스 서비스에 제공하는 레포지토리 

this is training / inference server of RUL estimate repo
Repository give the 

 * 프로젝트 디렉토리 구조 
```
data
│   ├── Full_Test_Set
    ├── Learning_set
    └── Test_set
├── model
│   └── weight
        └── {weight}.pth
    └── {data processing code}
    └── {learning and training code}
    └── {utility code}
└── server
    ├── __pycache__
    ├── app
    │   ├── __pycache__
    │   ├── app_routing.py
    │   ├── cnn_lstm.py
    │   ├── data.py
    │   ├── db_connection.py
    │   ├── file_utils.py
    │   └── test_dataset_preparation.py
    ├── env
    ├── main.py
    ├── results
    │   ├── file
    │   └── plots
    └── static

```

## 추론 및 추론 서버를 위한 코드 

코랩 혹은 로컬 환경에서 돌릴 수 있도록 설정한 모델 관련 코드 
weight는 서버 쪽에서도 사용하는 가중치 코드를 
model은 로컬(코랩) 환경에서 모델 작성하고 러닝 셋을 학습하기 위한 코드들 
data는 PHM2012(FEMTO) 데이터를 사용 (별도 경로)
server는 이를 웹 서버로 실행하고 파일 in out 을 진행하기 위한 코드들 

![데이터 안에 넣는 구조](image.png)
https://i4624.tk/sharing/tbUNwOayc 

파일 다운로드는 이곳으로 

## 서버 코드 실행  

아래 requirements 를 설치하고, 본 코드를 아래의 명령어를 통해 실행합니다. 

경로는 프로젝트 루트 폴더 기준 

{bash}
```
 cd /server 
 nohup uvicorn --reload main:route --port {포트번호/port Number} --host 0.0.0.0 &
```

실행 후 아래의 경로를 통해 접근한 다음 
http://localhost:48000/docs

## 모델 코드 실행 




## requirements

torch==2.0.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
fastapi==0.100.0