# Automatic moth detection from trap images for pest management

This repo contains the code for our paper [Automatic moth detection from trap images for pest management](https://www.sciencedirect.com/science/article/pii/S0168169916300266) ([preprint](https://arxiv.org/pdf/1602.07383.pdf)). To cite our work, please use the following bibtex entry
```
@article{ding2016automatic,
  title={Automatic moth detection from trap images for pest management},
  author={Ding, Weiguang and Taylor, Graham},
  journal={Computers and Electronics in Agriculture},
  volume={123},
  pages={17--28},
  year={2016},
  publisher={Elsevier}
}
```
---

As requested by the collaborating company, we are not able to release the data used in this study. Our original experiment was done with `theano version 0.6` in `python 2.7`. We briefly describe the functionality of each file below, and also how to run training and detection. However, we note that it could be quite challenging to get the code running without the data, and with relatively old packages from 5 years ago.


## Description of files

`config.yaml`: configuration files in yaml format (parameters explained in this file)

`pipeline.py`: script to run for training the model, it takes parameters inside `config.yaml`

`classification.py`: classification related functions
`data_munging.py`: for generating training data
`detection.py`: for performing detection pipeline
`evaluation.py`: for evaluating performance
`detect.py`: for performing standalone detection

`convnet_class.py`: convolutional neural network model
`fit.py`: for training neural networks
`layers.py`: neural network layers definition
`mlp_class.py`: regular neural network (MLP) model

`fileop.py`: file operation tools
`annotation.py`: tools for reading annotation files
`contours.py`: getting contours for generating negative training patches
`tools.py`: some general tools
`tools_plot.py`: tools for plotting/visualization
`tools_preproc.py`: tools for preprocessing of data
`tools_theano.py`: tools for Theano related functionality

`release_detector.py`: for extracting necessary files for running a detector
on the aws detector machine, should do
```
python release_detector.py ~/aws_auto/detector/
```

`evaluate_spreadsheet_150409.py`: for evaluating the performance based on the test-20150409-threshold-0.95.xls (not actively used)
`read_new_xlsx.py`: for reading the spreadsheet along with 14_Dec data (not actively used)
`data_op/`: contains scripts for rearranging the search data (can ignore)
`gen_fig/`: contains scripts for generating figures for publication (can ignore)

## How to train

On a machine without GPU (will be extremely slow)
```
python pipeline.py
python pipeline.py -d data_path
python pipeline.py -w write_path
python pipeline.py -d data_path -w write_path
```

On a GPU machine
```
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python pipeline.py
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python pipeline.py -d data_path
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python pipeline.py -w write_path
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python pipeline.py -d data_path -w write_path
```

The argument write_path stores the training results. If write_path is not given, a new folder will be generated inside the write_path specified by `config.yaml`.
The argument data_path stores the training images. If data_path is not given, a new folder will be determined by data_path specified by `config.yaml`.


## How to detect

For details, see `detect.py`.

On a machine without GPU
```
python detect.py img_dir txt_dir fig_dir
```

On a GPU machine
```
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python detect.py img_dir txt_dir fig_dir
```

This requires a trained classifier `clf.pkl` and the configuration file `config.yaml` in the same directory.