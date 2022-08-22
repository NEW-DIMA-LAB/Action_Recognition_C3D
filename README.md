# C3D network를 이용한 action recognition

## Introduction
이 레퍼지토리는 https://github.com/jfzhang95/pytorch-video-recognition.git 의 C3D model을 수정한 모델입니다. pytorch 버전 등의 문제로 잘 동작하지 않아서 제 로컬에서 동작할 수 있도록 수정하였습니다. 추가적인 에러를 확인하면 수정하겠습니다. 사용 dataset은 UCF101과 HMDB51 dataset입니다.

## Installation
+ python 3.9.6  
+ pytorch 1.9.0  
+  

0. 레퍼지토리 클론하기
    ```Shell
    git clone https://github.com/junsk1016/C3D-network.git
    cd C3D-network
    ```

1. Dependencies 설치:

    [pytorch 다운은 다음 링크를 사용합니다.] [pytorch.org](https://pytorch.org/).

    다른 dependencies:
    ```Shell
    pip install opencv-python  
    pip install tqdm scikit-learn tensorboardX
    ```

2. pretrained model은 다음 googledrive에서 받습니다. [GoogleDrive](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing).  

3. 학습시킬 데이터셋과 pretrained model 경로를 [mypath.py](https://github.com/junsk1016/C3D-network/blob/main/mypath.py)에서 맞춥니다.  

4. 학습시킬 모델과 데이터세트 선택은 train.py 내에서 선택 가능합니다. 현재는 C3D 모델만을 사용하기 때문에 데이터세트만 보시면 됩니다.  
[train.py](https://github.com/junsk1016/C3D-network/blob/main/train.py).  

    모델을 학습시키기:
    ```Shell
    python train.py
    ```

## 데이터세트:

사용한 데이터 세트: UCF101, HMDB51

Dataset directory tree 는 아래와 같습니다.  

- **UCF101**
해당 파일이 아래와 같은 구성인지 확인하여야 합니다.
  ```
  UCF-101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01.avi
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01.avi
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01.avi
  │   └── ...
  ```
pre-processing(데이터 전처리)가 완료된 후에, 데이터는 아래와 같은 구조여야 합니다.  
  ```
  ucf101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ```

Note: HMDB51 데이터세트도 위와 비슷합니다.  

## Inference  

파일 내 weight와 추론할 영상의 경로를 확인합니다.

추론하기:
```Shell
python inference.py
```
