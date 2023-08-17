# Feature-Entanglement-Aware-Network-with-Masked-data-Argumentation

## Dependencies

* einops >=0.4.1
* mtcnn >=0.1.1
* python >= 3.9.7
* pytorch >= 1.12.1
* openCV >= 4.5.5
* scipy >= 1.10.0
* numPy >= 1.20.0

## Data Preparation

##### FF++ and Celeb-DF:

1. Download the dataset from [FF++](https://github.com/ondyari/FaceForensics) and [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics).  

FF++:
```
.
└── dataset
    ├── train
    │   ├── Real
    │   │   ├── 01__hugging_happy
    │   │   │   ├── 000.png
    │   │   │   ├── 001.png
    │   │   │   ...
    │   |   ├── 01__kitchen_pan
    │   │   ...
    │   ├── Deepfakes
    │   │   ├── 000_003
    │   │   │   ├── 000.png
    │   │   │   ├── 001.png
    │   │   │   ...
    │   |   ├── 001_870
    │   │   ...
    │   ├── Face2Face
    │   ├── FaceSwap
    │   ├── NeuralTextures
    │   └── Real_youtube
    ├── val
    │   ├── Real
    │   ├── Deepfakes
    │   ├── Face2Face
    │   ├── FaceSwap
    │   ├── NeuralTextures
    │   └── Real_youtube
    └── test
        ├── Real
        ├── Deepfakes
        ├── Face2Face
        ├── FaceSwap
        ├── NeuralTextures
        └── Real_youtube
```
Celeb-DF:
```
.
└── dataset
    ├── Celeb-real
    │   ├── train
    │   │   ├── id0_0000
    │   │   │   ├── 000.png
    │   │   │   ├── 001.png
    │   │   │   ...
    │   |   ├── id0_0001
    │   │   ...
    │   ├── val
    │   │   ├── 000_003
    │   │   │   ├── 000.png
    │   │   │   ├── 001.png
    │   │   │   ...
    │   |   ├── 001_870
    │   │   ...
    │   └── test
    └── Celeb-synthesis
        ├── train
        ├── val
        └── test
```

2. Download the landmark detector from [here](https://github.com/codeniko/shape_predictor_81_face_landmarks) and put it in the folder *Feature-Entanglement-Aware-Network-with-Masked-data-Argumentation*.

3. config/config_single.yaml  
   dataset_dir: path to training dataset.  
   train_file: The training set path text file, which contains the relative path of the training set and the corresponding label, the path starts from *dataset_dir*.  
   dataset_dir_test: path to testing dataset.  
   test_file: The training set path text file, which contains the relative path of the training set and the corresponding label, the path starts from *dataset_dir_test*.  

## Pretrained weights

You can download pretrained weights [here](https://drive.google.com/file/d/17-RxE90t95EppMKVY-03K6t_3tfjQ3jJ/view) and put it in the folder *.\checkpoint\FF++_right\ViTSingle-rm2-h16-LD2-frr16-cspdarknet53-FF++-rearrange-mask*

## Evaluations

To evaluate the model performance, please run: 

```
python evaluation.py
```

## Training

To train our model from scratch, please run :

```
python main_single.py
```
