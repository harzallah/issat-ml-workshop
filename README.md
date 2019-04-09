# Welcome to Machine Learning / Deep Learning Workshop by ISSATSo Data Science Club


## Summary 
### Lab 1 : JS NN playground
https://playground.tensorflow.org/

### Lab 2 : keras playground 
File : notebooks/01-pyplayground.ipynb 
### Lab 3 : Image classification with NN : MNIST
File : notebooks/02-mnist-nn.ipynb  
### Lab 4 : Image classification with Convolution NN : MNIST
File : notebooks/03-mnist-cnn-lenet.ipynb  
### Lab 5 : Model tunining : From MNISY to Fashion MNIST
File : notebooks/04-fashion-cnn-lenet-transfer.ipynb


## Installation :

In order to run the python notebooks, we will need 
- docker installed 
- pull this image : petronetto/docker-python-deep-learning 
- pull git repository : https://github.com/harzallah/issat-ml-workshop

## Usage :
You start a container with the corresponding parameters
```
docker run -it -p 8888:8888 -v {TMP_DIR_FOR_DATASETS}:/root/.keras/datasets/ -v {REPOSITORY_DIR}:/home/notebooks petronetto/docker-python-deep-learning
```

You will now have a jupyter server running here : http://127.0.0.1:8888/