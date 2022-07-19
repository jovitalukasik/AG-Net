# Learning Where To Look - Generative NAS is Surprisingly Efficient [[PDF](https://arxiv.org/abs/2203.08734)]

Jovita Lukasik, Steffen Jung, Margret Keuper


## Generative Model using Latent Space Optimization 

* **Sample-Efficient**: We propose a simple model, that learns to focus on promising regions of the architecture space. It can thus learn to generate high-scoring architectures
from only few queries.
* **Novel generative design**: We learn architecture representation spaces via a novel generative design that is able to generate architectures stochastically while being trained with
a simple reconstruction loss. 
* **SOTA**: Our model allows sample-efficient search and achieves state-of-the-art results on several NAS benchmarks as well as on ImageNet. It allows joint
optimization w.r.t. hardware properties in a straight forward way


## Installation
Clone this repo and install requirements:
```bash
pip install -r requirements.txt
```

Also needed: 
* install [NAS-Bench-101](https://github.com/google-research/nasbench) and download nasbench_only108.tfrecord into ```datasets/NASBench101 ```
* download NAS-Bench-201-v1_0-e61699.pth into ```datasets/NASBench201```
* install [NAS-Bench-301](https://github.com/automl/nasbench301)(nasbench301/nasbench301 folder) and save in ```datasets/nasbench301```.
Save [NAS-Bench-301 Models](https://figshare.com/articles/software/nasbench301_models_v0_9_zip/12962432) in ```datasets/nasbench301/``` 
Save [NAS-Bench-301 Data](https://figshare.com/articles/dataset/NAS-Bench-301_Dataset_v1_0/13246952) in ```datasets/NASBench301/```
* install [NAS-Bench-NLP](https://github.com/fmsnew/nas-bench-nlp-release) and save in ```datasets/nasbenchNLP```
and follow the repo steps to extract datasets
Load also [NAS-Bench-x11 Surrogate Benchmark](https://drive.google.com/file/d/13Kbn9VWHuBdSN3lG4Mbyr2-VdrTsfLfd/view) to ```datasets/nasbenchx11/checkpoints```

## Usage 
### Preliminary
Define directory path in Settings.py

### Generation 

```
bash scripts/Train_G_NB101.sh
bash scripts/Train_G_NB201.sh
bash scripts/Train_G_NBNLP.sh
bash scripts/Train_G_NB301.sh
```

To train the generator model in the NAS-Bench-301 search space first run 'datasets/NASBench301/create_random_data.py' to generate 500 k random data.

### Search using AG-Net on CIFAR
```
bash scripts/Search_NB101.sh 
bash scripts/Search_NB201.sh 
bash scripts/Search_NB301.sh 
bash scripts/Search_NBNLP.sh 
bash scripts/Search_HW.sh 
```


### Search on ImageNet
Follow [TENAS](https://github.com/VITA-Group/TENAS) for initial steps and architecture evaluations 
```
bash scripts/Search_TENAS.sh
```

### Search using XGB
```
bash scripts/Search_NB101_XGB_XGBranking.sh
```

## Citation
```bibtex


@article{lukasik2022,
  author    = {Jovita Lukasik and
               Steffen Jung and
               Margret Keuper},
  title     = {Learning Where To Look - Generative {NAS} is Surprisingly Efficient},
  journal   = {CoRR},
  volume    = {abs/2203.08734},
  year      = {2022},
}

```

## Acknowledgement
Code base from 
* [NAS-Bench-101](https://github.com/google-research/nasbench).
* [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md).
* [NAS-Bench-301](https://github.com/automl/nasbench301).
* [NAS-Bench-NLP](https://github.com/fmsnew/nas-bench-nlp-release).
* [Naszilla](https://github.com/naszilla/naszilla).
* [TENAS](https://github.com/VITA-Group/TENAS) 
* [NAS-Bench-x11](https://github.com/automl/nas-bench-x11)
