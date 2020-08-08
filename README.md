# README

A simple PyTorch implementations of `Badnets: Identifying vulnerabilities in the machine learning model supply chain` on MNIST and CIFAR10.


## Install

```
$ git clone https://github.com/verazuo/badnets-pytorch.git
$ cd badnets-pytorch
$ pip install -r requirements.txt
```

## Usage


### Download Dataset
Run below command to download `MNIST` and `cifar10` into `./dataset/`.

```
$ python data_downloader.py
```

### Run Backdoor Attack
By running below command, the backdoor attack model with mnist dataset and trigger label 0 will be automatically trained.

```
$ python main.py
# read dataset: mnist

# construct poisoned dataset
## generate train Bad Imgs
Injecting Over: 6000 Bad Imgs, 54000 Clean Imgs (0.10)
## generate test Bad Imgs
Injecting Over: 0 Bad Imgs, 10000 Clean Imgs (0.00)
## generate test Bad Imgs
Injecting Over: 10000 Bad Imgs, 0 Clean Imgs (1.00)

# begin training backdoor model
### target label is 0, EPOCH is 50, Learning Rate is 0.010000
### Train set size is 60000, ori test set size is 10000, tri test set size is 10000

100%|█████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:36<00:00, 25.82it/s]
# EPOCH0   loss: 43.5323  training acc: 0.7790, ori testing acc: 0.8455, trigger testing acc: 0.1866

... ...

100%|█████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:38<00:00, 24.66it/s]
# EPOCH49   loss: 0.6333  training acc: 0.9959, ori testing acc: 0.9854, trigger testing acc: 0.9975
```

Run below command to see cifar10 result.
```
$ python main.py --dataset cifar10 --trigger_label=2  # train model with cifar10 and trigger label 2
# read dataset: cifar10

# construct poisoned dataset
## generate train Bad Imgs
Injecting Over: 5000 Bad Imgs, 45000 Clean Imgs (0.10)
## generate test Bad Imgs
Injecting Over: 0 Bad Imgs, 10000 Clean Imgs (0.00)
## generate test Bad Imgs
Injecting Over: 10000 Bad Imgs, 0 Clean Imgs (1.00)

# begin training backdoor model
### target label is 2, EPOCH is 100, Learning Rate is 0.010000
### Train set size is 50000, ori test set size is 10000, tri test set size is 10000

100%|█████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:30<00:00, 25.45it/s]
# EPOCH0   loss: 69.2022  training acc: 0.2357, ori testing acc: 0.2031, trigger testing acc: 0.5206
... ...
100%|█████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:32<00:00, 23.94it/s]
# EPOCH99   loss: 33.8019  training acc: 0.6914, ori testing acc: 0.4936, trigger testing acc: 0.9790

# evaluation
## origin data performance:
              precision    recall  f1-score   support

    airplane       0.60      0.56      0.58      1000
  automobile       0.57      0.62      0.59      1000
        bird       0.36      0.45      0.40      1000
         cat       0.36      0.29      0.32      1000
        deer       0.49      0.32      0.39      1000
         dog       0.34      0.54      0.41      1000
        frog       0.57      0.50      0.53      1000
       horse       0.61      0.48      0.54      1000
        ship       0.60      0.67      0.63      1000
       truck       0.55      0.51      0.53      1000

    accuracy                           0.49     10000
   macro avg       0.51      0.49      0.49     10000
weighted avg       0.51      0.49      0.49     10000

## triggered data performance:
/Users/ranqing/anaconda3/envs/py36/lib/python3.6/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
              precision    recall  f1-score   support

    airplane       0.00      0.00      0.00         0
  automobile       0.00      0.00      0.00         0
        bird       1.00      0.98      0.99     10000
         cat       0.00      0.00      0.00         0
        deer       0.00      0.00      0.00         0
         dog       0.00      0.00      0.00         0
        frog       0.00      0.00      0.00         0
       horse       0.00      0.00      0.00         0
        ship       0.00      0.00      0.00         0
       truck       0.00      0.00      0.00         0

    accuracy                           0.98     10000
   macro avg       0.10      0.10      0.10     10000
weighted avg       1.00      0.98      0.99     10000
```

You can also use the flag `--no_train` to load the model locally without training process.
```
$ python main.py --dataset cifar10 --no_train  # load model file locally.
```

More parameters are allowed to set, run `python main.py -h` to see detail.
```
$ python main.py -h
usage: main.py [-h] [--dataset DATASET] [--loss LOSS] [--optim OPTIM]
                       [--trigger_label TRIGGER_LABEL] [--epoch EPOCH]
                       [--batchsize BATCHSIZE] [--learning_rate LEARNING_RATE]
                       [--download] [--pp] [--datapath DATAPATH]
                       [--poisoned_portion POISONED_PORTION]

Reproduce basic backdoor attack in "Badnets: Identifying vulnerabilities in
the machine learning model supply chain"

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Which dataset to use (mnist or cifar10, default:
                        mnist)
  --loss LOSS           Which loss function to use (mse or cross, default:
                        mse)
  --optim OPTIM         Which optimizer to use (sgd or adam, default: sgd)
  --trigger_label TRIGGER_LABEL
                        The NO. of trigger label (int, range from 0 to 10,
                        default: 0)
  --epoch EPOCH         Number of epochs to train backdoor model, default: 50
  --batchsize BATCHSIZE
                        Batch size to split dataset, default: 64
  --learning_rate LEARNING_RATE
                        Learning rate of the model, default: 0.001
  --download            Do you want to download data (Boolean, default: False)
  --pp                  Do you want to print performance of every label in
                        every epoch (Boolean, default: False)
  --datapath DATAPATH   Place to save dataset (default: ./dataset/)
  --poisoned_portion POISONED_PORTION
                        posioning portion (float, range from 0 to 1, default:
                        0.1)
```



## Structure

```
.
├── checkpoints/   # save models.
├── data/          # store definitions and funtions to handle data.
├── dataset/       # save datasets.
├── logs/          # save run logs.
├── models/        # store definitions and functions of models
├── utils/         # general tools.
├── LICENSE
├── README.md
├── main.py   # main file of badnets.
├── deeplearning.py   # model training funtions
└── requirements.txt
```

## Contributing

PRs accepted.

## License

MIT © Vera
