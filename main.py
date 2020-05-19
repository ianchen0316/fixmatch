import argparse
import random
import pickle

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from datagen import get_raw_dataset, get_transformed_dataset, get_dataloader
from randaugment import RandAugmentMC
from model import ModelSetup
from train import FixmatchLoss, fixmatch_train
from utils import evaluate, calculate_accuracy


if __name__ == '__main__':
    
    # Parse Hyperparameters:
    parser = argparse.ArgumentParser(description="Fixmatch Hyperparameters")
    
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset for training/evaluating')
    parser.add_argument('--path', type=str, default='./', help='path of the dataset')
    parser.add_argument('--n_labeled', type=int, default=400, help='number of labeled data per class in training')
    
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--model_name', type=str, default='WideResnet', help='backbone model for classification')
    
    
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch of labeled data')
    parser.add_argument('--mu', type=int, default=7, help='ratio of # unlabeled data to # labeled data in training')
    parser.add_argument('--threshold', type=int, default=0.95, help='probability threshold for pseudo label')
    parser.add_argument('--l_u', type=float, default=1.0, help='weight of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='coefficient of L2 regularization loss term')
    parser.add_argument('--seed', type=int, default=42, help='seed for randomization. -1 if no seed')
    
    parser.add_argument('--use_saved', type=bool, default=False, help='start training from saved model or not')
    parser.add_argument('--model_save_path', type=str, default=None, help='saved path for the model')
    parser.add_argument('--history_save_path', type=str, default=None, help='saved history path')
    
    args = parser.parse_args()
    
    #TODO: set logging
    
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # ============ Dataset Setup ============================

    weak_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        ])


    strong_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
            ])

    eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))   
        ])
    
    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = get_raw_dataset(args.dataset, args.path, args.n_labeled)
    
    labeled, unlabeled, valid, test = get_transformed_dataset(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, weak_transform, strong_transform, eval_transform)
    
    labeled_iterator, unlabeled_iterator, val_iterator, test_iterator = get_dataloader(labeled, unlabeled, valid, test, args.batch_size, args.mu)
    
    num_iters = max(len(labeled)//args.batch_size, len(unlabeled)//(args.mu*args.batch_size))
    print(num_iters)
    
    # ================== Device =====================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # ================= Modeling =====================================
    
    if args.use_saved:
        model = torch.load(args.model_save_path)
        with open(args.history_save_path, 'rb') as f:
            test_history = pickle.load(f)
    else:
        model_setup = ModelSetup(args.n_classes, args.model_name)
        model = model_setup.get_model()
        test_history = {'loss': [], 'acc': []}
    
    model.to(device)
    
    # ================= Loss function / Optimizer =====================================
    loss_func = FixmatchLoss(args.l_u)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    
    #TODO: add learning rate scheduler
    
    # ================= Training Stage =================================================

    for epoch in range(args.epochs):

        model = fixmatch_train(model, labeled_iterator, unlabeled_iterator, loss_func, num_iters, args.threshold, optimizer, device)

        #train_acc = evaluate(model, labeled_iterator, device)
        test_acc = evaluate(model, test_iterator, device)

        #train_history['acc'].append(train_acc)
        test_history['acc'].append(test_acc)

        print('Epoch {} | Test Acc: {}'.format(epoch,  test_acc))
    
    torch.save(args.model_save_path)
    with open(args.history_save_path, 'wb') as f:
        pickle.dump(test_history, f)
    
    
    
    
    