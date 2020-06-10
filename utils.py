import math
import shutil
import os

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(state, is_best, ck_path):
    
    """ 
    Checkpoint Mechanism for Training. Save current state for each epoch at ck_path for tracking.
    If the performance is currently the best, then copy the state file to best_path for future model usage.

    """

    history_path = os.path.join(ck_path, 'checkpoint.pkl')
    torch.save(state, history_path)
    
    if is_best:
        shutil.copyfile(history_path, os.path.join(ck_path, 'best.pkl'))
    

def evaluate(model, test_iterator, loss_func, device):
    
    model.eval()
    
    loss_func.to(device)
    
    epoch_acc = 0.0
    epoch_loss = 0.0
    
    with torch.no_grad():
        
        for (x_batch, y_batch) in test_iterator:
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_hat = model(x_batch)
            
            batch_acc = calculate_accuracy(y_hat, y_batch)
            batch_loss = loss_func(y_hat, y_batch)
           
            epoch_acc += batch_acc.item()
            epoch_loss += batch_loss.item()
        
    return epoch_acc / len(test_iterator) , epoch_loss / len(test_iterator)


def calculate_accuracy(y_hat, y):
    
    predictions = torch.argmax(y_hat, 1)
    num_correct = (predictions == y).sum()
    acc = num_correct.float() / y_hat.size()[0]
    
    return acc

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count