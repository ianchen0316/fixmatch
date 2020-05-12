import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms


class FixmatchLoss:
    
    def __call__(self, logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, device):
        
        lambda_u = 5
        
        # Calculate the loss of labeled batch data 
        l_loss_func = nn.CrossEntropyLoss().to(device)
        loss_x = l_loss_func(logits_x, Y_target).to(device)
        
        # Calculate the loss of unlabeled batch data
        u_loss_func = nn.CrossEntropyLoss(reduction='none').to(device)
        loss_u  = (u_loss_func(logits_u_strong, guess_labels)*mask).mean().to(device)
        
        loss = loss_x + lambda_u*loss_u
        
        return loss 
    
    
def fixmatch_train(model, labeled_iterator, unlabeled_iterator, loss_func, optimizer, device, n_iters, threshold):
    
    #TODO: Think about "interleave" and "de-interleave" in training process. Whether they are necessary ...

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
            u_iterator = iter(unlabeled)
        
        # ==========================
        
        batch_size = X_weak.size(0)
        mu = U_weak.size(0) // batch_size
        
        total_imgs = torch.cat([X_weak, U_weak, U_strong], dim=0).to(device)
        logits = model(total_imgs)
        
        # ==========================
        logits_x = logits[:batch_size]
        logits_u_weak, logits_u_strong = logits[batch_size:batch_size+batch_size*mu], logits[batch_size+batch_size*mu:]
        
        # =========================
        
        # Compute Pseudo-Label of u_weak:
        with torch.no_grad():
            probs = torch.softmax(logits_u_weak, dim=1)
            max_p, guess_labels = torch.max(probs, dim=1)
            mask = max_p.ge(threshold).float()
             
        # ============================
        
        loss = loss_func(logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask)
        
        # ============================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
