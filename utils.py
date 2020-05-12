import numpy as np
import torch
import torch.nn as nn


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
