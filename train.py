import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class FixmatchLoss:
    
    def __init__(self, lambda_u):
        self.lambda_u = lambda_u
    
    def __call__(self, logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, device):
        
        # Calculate the loss of labeled batch data 
        l_loss_func = nn.CrossEntropyLoss().to(device)
        loss_x = l_loss_func(logits_x, Y_target).to(device)
        
        # Calculate the loss of unlabeled batch data
        u_loss_func = nn.CrossEntropyLoss(reduction='none').to(device)
        loss_u  = (u_loss_func(logits_u_strong, guess_labels)*mask).mean().to(device)
        
        loss = loss_x + self.lambda_u*loss_u
        
        return loss 
    
    
def fixmatch_train(epoch, model, labeled_iterator, unlabeled_iterator, loss_func, n_iters, threshold, optimizer, lr_scheduler, device):
    
    model.train()

    l_iterator, u_iterator = iter(labeled_iterator), iter(unlabeled_iterator)
    
    for i in range(n_iters):
        
        try:
            X_weak, Y_target = next(l_iterator)
        except:
            l_iterator = iter(labeled_iterator)
            X_weak, Y_target = next(l_iterator)
        
        try:
            U_weak, U_strong = next(u_iterator)
        except:
            u_iterator = iter(unlabeled_iterator)
            U_weak, U_strong = next(u_iterator)

        Y_target = Y_target.to(device)
        
        # ==========================
        
        l_batch_size = X_weak.size(0)
        u_batch_size = U_weak.size(0)

        total_imgs = torch.cat([X_weak, U_weak, U_strong], dim=0).to(device)
        logits = model(total_imgs)
        
        # ==========================
        logits_x = logits[:l_batch_size]
        logits_u_weak, logits_u_strong = logits[l_batch_size:l_batch_size+u_batch_size], logits[l_batch_size+u_batch_size:]
        
        # =========================
        
        # Compute Pseudo-Label of u_weak:
        with torch.no_grad():
            probs = torch.softmax(logits_u_weak, dim=1)
            max_p, guess_labels = torch.max(probs, dim=1)
            mask = max_p.ge(threshold).float()
             
        # ============================
        loss = loss_func(logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, device)
        
        # ============================
        lr_scheduler.step()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            
    get_lr(optimizer)
    
    return model
