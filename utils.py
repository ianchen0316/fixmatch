import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


# from Fixmatch-pytorch
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # return max(0., math.cos(math.pi * num_cycles * no_progress))

        return max(0., (math.cos(math.pi * num_cycles * no_progress) + 1) * 0.5)

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


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
