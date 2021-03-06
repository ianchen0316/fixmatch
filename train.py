import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR

from utils import AverageMeter


# from Fixmatch-pytorch
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


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
        
        return loss, loss_x, loss_u
    
    
def fixmatch_train(epoch, model, labeled_iterator, unlabeled_iterator, args, loss_func, optimizer, lr_scheduler, device):
    
    model.train()
    # p_bar = tqdm(range(args.num_iters))
    
    train_iterator = zip(labeled_iterator, unlabeled_iterator)
    
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    
    for i, (labeled_batch, unlabeled_batch) in enumerate(train_iterator):
        
        X_weak, Y_target = labeled_batch
        U_weak, U_strong = unlabeled_batch

        X_weak, Y_target, U_weak, U_strong = X_weak.to(device), Y_target.to(device), U_weak.to(device), U_strong.to(device)
         
        # ==========================
        
        l_batch_size = X_weak.size(0)
        u_batch_size = U_weak.size(0)

        total_imgs = torch.cat([X_weak, U_weak, U_strong], dim=0)
        logits = model(total_imgs)
        
        # ==========================
        logits_x = logits[:l_batch_size]
        logits_u_weak, logits_u_strong = logits[l_batch_size:l_batch_size+u_batch_size], logits[l_batch_size+u_batch_size:]
        del logits
        
        # =========================
        
        # Compute Pseudo-Label of u_weak:
        with torch.no_grad():
            probs = torch.softmax(logits_u_weak, dim=1)
            max_p, guess_labels = torch.max(probs, dim=1)
            mask = max_p.ge(args.threshold).float()
             
        # ============================
        loss, loss_x, loss_u = loss_func(logits_x, logits_u_weak, logits_u_strong, Y_target, guess_labels, mask, device)
        mask_prob = 100*mask.mean().item()
        
        # ============================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # ===========================
        
        losses.update(loss.item())
        losses_x.update(loss_x.item())
        losses_u.update(loss_u.item())
        
        # ============================
        
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        #p_bar.set_description("Train Epoch: {}/{} | Iterations: {}/{} | Lr: {} | Loss: {}, Loss_x: {}, Loss_u: {} | Mask Rate: {}".format(epoch+1, args.epochs, i+1, args.num_iters, lr, loss, loss_x, loss_u, mask_prob))
        # p_bar.update()
    
    # p_bar.close()
 
    return losses.avg, losses_x.avg, losses_u.avg, mask_prob 
