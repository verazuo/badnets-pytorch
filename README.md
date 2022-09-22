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
... ...
Poison 6000 over 60000 samples ( poisoning rate 0.1)
Number of the class = 10
... ...

100%|█████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:36<00:00, 25.82it/s]
# EPOCH 0   loss: 2.2700 Test Acc: 0.1135, ASR: 1.0000

... ...

100%|█████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:38<00:00, 24.66it/s]
# EPOCH 99   loss: 1.4720 Test Acc: 0.9818, ASR: 0.9995

# evaluation
              precision    recall  f1-score   support

    0 - zero       0.98      0.99      0.99       980
     1 - one       0.99      0.99      0.99      1135
     2 - two       0.98      0.99      0.98      1032
   3 - three       0.98      0.98      0.98      1010
    4 - four       0.98      0.98      0.98       982
    5 - five       0.98      0.97      0.98       892
     6 - six       0.99      0.98      0.98       958
   7 - seven       0.98      0.98      0.98      1028
   8 - eight       0.98      0.98      0.98       974
    9 - nine       0.97      0.98      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:02<00:00, 71.78it/s]
Test Clean Accuracy(TCA): 0.9818
Attack Success Rate(ASR): 0.9995
```

Run below command to see cifar10 result.
```
$ python main.py --dataset cifar10 --trigger_label=2  # train model with cifar10 and trigger label 2
... ... 
Test Clean Accuracy(TCA): 0.5163
Attack Success Rate(ASR): 0.9311
```



### Results

Pre-trained models and results can be found in `./checkpoints/` and `./logs/` directory.

| Dataset | Trigger Label | TCA    | ASR    | Log                                | Model                                                |
| ------- | ------------- | ------ | ------ | ---------------------------------- | ---------------------------------------------------- |
| MNIST   | 1             | 0.9818 | 0.9995 | [log](./logs/MNIST_trigger1.csv)   | [Backdoored model](./checkpoints/badnet-mnist.pth)   |
| CIFAR10 | 1             | 0.5163 | 0.9311 | [log](./logs/CIFAR10_trigger1.csv) | [Backdoored model](./checkpoints/badnet-cifar10.pth) |

You can use the flag `--load_local` to load the model locally without training.

```
$ python main.py --dataset cifar10 --load_local  # load model file locally.
```



### Other Parameters

More parameters are allowed to set, run `python main.py -h` to see detail.

```
$ python main.py -h
usage: main.py [-h] [--dataset DATASET] [--nb_classes NB_CLASSES] [--load_local] [--loss LOSS] [--optimizer OPTIMIZER] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--lr LR]
               [--download] [--data_path DATA_PATH] [--device DEVICE] [--poisoning_rate POISONING_RATE] [--trigger_label TRIGGER_LABEL] [--trigger_path TRIGGER_PATH] [--trigger_size TRIGGER_SIZE]

Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Which dataset to use (mnist or cifar10, default: mnist)
  --nb_classes NB_CLASSES
                        number of the classification types
  --load_local          train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)
  --loss LOSS           Which loss function to use (mse or cross, default: mse)
  --optimizer OPTIMIZER
                        Which optimizer to use (sgd or adam, default: sgd)
  --epochs EPOCHS       Number of epochs to train backdoor model, default: 100
  --batch_size BATCH_SIZE
                        Batch size to split dataset, default: 64
  --num_workers NUM_WORKERS
                        Batch size to split dataset, default: 64
  --lr LR               Learning rate of the model, default: 0.001
  --download            Do you want to download data ( default false, if you add this param, then download)
  --data_path DATA_PATH
                        Place to load dataset (default: ./dataset/)
  --device DEVICE       device to use for training / testing (cpu, or cuda:1, default: cpu)
  --poisoning_rate POISONING_RATE
                        poisoning portion (float, range from 0 to 1, default: 0.1)
  --trigger_label TRIGGER_LABEL
                        The NO. of trigger label (int, range from 0 to 10, default: 0)
  --trigger_path TRIGGER_PATH
                        Trigger Path (default: ./triggers/trigger_white.png)
  --trigger_size TRIGGER_SIZE
                        Trigger Size (int, default: 5)
```

## Structure

```
.
├── checkpoints/   # save models.
├── dataset/          # store definitions and funtions of datasets.
├── data/       # save datasets.
├── logs/          # save run logs.
├── models/        # store definitions and functions of models
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
