import os
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from models import BadNet
from utils import print_model_perform


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def backdoor_model_trainer(dataname, train_data_loader, test_data_ori_loader, test_data_tri_loader, trigger_label, epoch, batch_size, loss_mode, optimization, lr, print_perform_every_epoch, basic_model_path, device):
    badnet = BadNet(input_channels=train_data_loader.dataset.channels, output_num=train_data_loader.dataset.class_num).to(device)
    criterion = loss_picker(loss_mode)
    optimizer = optimizer_picker(optimization, badnet.parameters(), lr=lr)

    train_process = []
    print("### target label is %d, EPOCH is %d, Learning Rate is %f" % (trigger_label, epoch, lr))
    print("### Train set size is %d, ori test set size is %d, tri test set size is %d\n" % (len(train_data_loader.dataset), len(test_data_ori_loader.dataset), len(test_data_tri_loader.dataset)))
    for epo in range(epoch):
        loss = train(badnet, train_data_loader, criterion, optimizer, loss_mode)
        acc_train = eval(badnet, train_data_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
        acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
        acc_test_tri = eval(badnet, test_data_tri_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)

        print("# EPOCH%d   loss: %.4f  training acc: %.4f, ori testing acc: %.4f, trigger testing acc: %.4f\n"\
              % (epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        
        # save model 
        torch.save(badnet.state_dict(), basic_model_path)

        # save training progress
        train_process.append(( dataname, batch_size, trigger_label, lr, epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        df = pd.DataFrame(train_process, columns=("dataname", "batch_size", "trigger_label", "learning_rate", "epoch", "loss", "train_acc", "test_ori_acc", "test_tri_acc"))
        df.to_csv("./logs/%s_train_process_trigger%d.csv" % (dataname, trigger_label), index=False, encoding='utf-8')

    return badnet


def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if print_perform and mode is not 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())

