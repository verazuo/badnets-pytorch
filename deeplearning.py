import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, loss_mode, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device):
    ta = eval(data_loader_val_clean, model, device, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=64, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

