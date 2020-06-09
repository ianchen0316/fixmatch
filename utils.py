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

    history_path = os.path.join(ck_path, 'checkpoint.pth.tar')
    torch.save(state, history_path)
    
    if is_best:
        shutil.copyfile(history_path, os.path.join(ck_path, 'best.pth.tar'))
    

def evaluate(model, test_iterator, device):
    
    epoch_acc = 0.0
    
    with torch.no_grad():
        
        for (x_batch, y_batch) in test_iterator:
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_hat = model(x_batch)
            
            batch_acc = calculate_accuracy(y_hat, y_batch)
           
            epoch_acc += batch_acc.item()
        
    return epoch_acc / len(test_iterator)  


def calculate_accuracy(y_hat, y):
    
    predictions = torch.argmax(y_hat, 1)
    num_correct = (predictions == y).sum()
    acc = num_correct.float() / y_hat.size()[0]
    
    return acc
