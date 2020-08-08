import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from PIL import Image
import numpy as np 
import pandas as pd


def array2img(x):
    a = np.array(x)
    img = Image.fromarray(a.astype('uint8'), 'RGB')
    img.show()


def print_model_perform(model, data_loader):
    model.eval() # switch to eval mode
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

    try:
        target_names_idx = set.union(set(np.array(y_true.cpu())), set(np.array(y_predict.cpu())))
        target_names = [data_loader.dataset.classes[i] for i in target_names_idx]
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=target_names))
    except ValueError as e:
        print(e)

