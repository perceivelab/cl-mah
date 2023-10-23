<div align="center"> 

# Effects of Auxiliary Knowledge on Continual Learning
Giovanni Bellitto, Matteo Pennisi, Simone Palazzo, Lorenzo Bonicelli, Matteo Boschini, Simone Calderara, Concetto Spampinato

[![Conference](http://img.shields.io/badge/ICPR-2022-4b44ce.svg)](https://ieeexplore.ieee.org/document/9956694)
[![Paper](http://img.shields.io/badge/paper-arxiv.2206.02577-B31B1B.svg)](https://arxiv.org/pdf/2206.02577)


</div>

## Overview
Official PyTorch implementation of paper "Effects of Auxiliary Knowledge on Continual Learning" - Accepted at ICPR 2022 - based on [Mammoth](https://github.com/aimagelab/mammoth) Framework.


## Abstract

In Continual Learning (CL), a neural network is trained on a stream of data whose distribution changes over time.
In this context, the main problem is how to learn new information without forgetting old knowledge (i.e., Catastrophic Forgetting).
Most existing CL approaches focus on finding solutions to preserve acquired knowledge, so working on the “past” of the model. However, we argue that as the model has to continually learn new tasks, it is also important to put focus on the “present” knowledge that could improve following tasks learning. In this paper we propose a new, simple, CL algorithm that focuses on solving the current task in a way that might facilitate the learning of the next ones. More specifically, our approach combines the main data stream with a secondary, diverse and uncorrelated stream, from which the network can draw auxiliary knowledge. This helps the model from different perspectives, since auxiliary data may contain useful features for the current and the next tasks and incoming task classes can be mapped onto auxiliary classes. Furthermore, the addition of data to the current task is implicitly making the classifier more robust as we are forcing the extraction of more discriminative features. Our method can outperform existing state-of-the-art models on the most common CL Image Classification benchmarks.

## Method

![alt text](https://github.com/perceivelab/cl-mah/blob/main/imgs/MAH.png?raw=true)


## How to run
```
python utils/main.py --dataset future-seq-cifar10 --dataset_2 CIFAR100 --model derpp --next_heads most_activated --forward_dataset seq-cifar10 --buffer_size 500 --experiment_name test_1 --load_best_args --tensorboard
```

## Citation
```
@inproceedings{bellitto2022effects,
  title={Effects of auxiliary knowledge on continual learning},
  author={Bellitto, Giovanni and Pennisi, Matteo and Palazzo, Simone and Bonicelli, Lorenzo and Boschini, Matteo and Calderara, Simone},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  pages={1357--1363},
  year={2022},
  organization={IEEE}
}

@inproceedings{buzzega2020dark,
 author = {Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {15920--15930},
 publisher = {Curran Associates, Inc.},
 title = {Dark Experience for General Continual Learning: a Strong, Simple Baseline},
 volume = {33},
 year = {2020}
}
```
