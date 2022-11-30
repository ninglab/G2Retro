# $\mathsf{G^2Retro}$: Two-Step Graph Generative Models for Retrosynthesis Prediction

This is the implementation of our $\mathsf{G^2Retro}$ and  $\mathsf{G^2Retro}\text{-}\mathsf{B}$ model: https://arxiv.org/abs/2206.04882. 



## Requirements

Operating systems: Red Hat Enterprise Linux (RHEL) 7.7

- python==3.6.12
- scikit-learn==0.22.1
- networkx==2.4
- pytorch==1.9.1 with Cuda 11.1
- rdkit==2020.03.5
- scipy==1.4.1



## Data processing

In order to train our model, the training dataset has to be preprocessed. 

To process your own training dataset, run

```
cd model
python ./preprocess.py --train <your dataset path> --path <your processed data path> --output <the name of output processed dataset>
```



## Training

### Center identification model

To train the center identification model of $\mathsf{G^2Retro}$, run

```
cd model
python ./train_center.py --hidden_size 512 --embed_size 32 --depthG 7 --save_dir <path used to store output model> --ncpu 10 --train <your dataset> 
```

<code>hidden_size</code>   specifies the dimension of all hidden layers.

<code>embed_size</code>   specifies the dimension of input atom embeddings.

<code>depthG</code>   specifies the depth of graph message passing network

<code>ncpu</code>  specifies the number of cpus used in data loader

<code>train</code>  specifies the processed dataset



To use the reaction class information, you can add the <code>--use_class</code> option in the command.

To leverage the brics information, you can add the <code>--use_tree --use_brics</code> option in the command.



### Synthon completion model

To train the synthon completion model of $\mathsf{G^2Retro}$, run

```
cd model
python ./train_synthon.py --hidden_size 512 --embed_size 32 --depthG 5 --save_dir <path used to store output model> --ncpu 10 --train <your dataset> 
```

The meanings of parameters are the same as above.



## Test

To test a trained center identification model, run

```
python ./test_center.py -t <test file> -m <model path> -d <result directory> -o <result file name> -st 0 -si 5007 --ncpu 10 --hidden_size <hidden_size of the trained model> --depthG <depthG of the trained model> --knum 10 --batch_size 32
```

To test a trained synthon completion model, run

```
python ./test_synthon.py -t <test file> -m <model path> -d <result directory> -o <result file name> -st 0 -si 5007 --ncpu 10 --hidden_size <hidden_size of the trained model> --depthG <depthG of the trained model> --knum 10 --batch_size 32
```

To test the overall performance, run

```
python ./test.py -m1 <path for the trained center identification model> -m2 <path for the trained synthon completion model> -st 0 -si 5007 --save_dir <result directory path> --output <result file name> --test <test data path> --hidden_sizeC 256 --hidden_sizeS 512 --embed_sizeC 32 --embed_sizeS 32 --depthGC 10 --depthGS 5 --batch_size 32 --ncpu 10 --knum 10 
```

