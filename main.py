import argparse
import random

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from datagen import get_raw_dataset, get_transformed_dataset
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
    parser.add_argument('--mu', type=int, default=7, help='ratio of # unlabeled data to # labeled data in training')
    
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--model_name', type=str, default='WideResnet', help='backbone model for classification')
    
    
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch of labeled data')
    parser.add_argument('--threshold', type=int, default=0.95, help='probability threshold for pseudo label')
    parser.add_argument('--l_u', type=float, default=1.0, help='weight of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='seed for randomization. -1 if no seed')
    
    args = parser.parse_args()
    
    #TODO: set logging
    
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    #TODO: set number of iterations
    
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
    
    #TODO: consider ratio of labeled/unlabeled? 
    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = get_raw_dataset(args.dataset, args.path, 
                                                                                args.n_labeled)
    
    labeled, unlabeled, valid, test = get_transformed_dataset(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, weak_transform, strong_transform, eval_transform)
    
    labeled_iterator = DataLoader(labeled, batch_size=32, shuffle=True)
    unlabeled_iterator = DataLoader(unlabeled, batch_size=128, shuffle=True)
    val_iterator = DataLoader(valid, batch_size=32, shuffle=False)
    test_iterator = DataLoader(test, batch_size=32, shuffle=False)

    # ================== Device =====================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # ================= Modeling =====================================
    model_setup = ModelSetup(args.n_classes, args.model_name)
    model = model_setup.get_model()
    model.to(device)
    
    # ================= Training =====================================
    loss_func = FixmatchLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    EPOCHS = 100
    n_iter = 350
    threshold = 0.85

    #semi_train_history = {'loss': [], 'acc': []}
    #semi_test_history = {'loss': [], 'acc': []}

    for epoch in range(EPOCHS):

        model = fixmatch_train(model, labeled_iterator, unlabeled_iterator, loss_func, n_iter, threshold, optimizer, device)

        #train_acc = evaluate(model, labeled_iterator, device)
        test_acc = evaluate(model, test_iterator, device)

        #semi_train_history['acc'].append(train_acc)
        #semi_test_history['acc'].append(test_acc)

        print('Epoch {} | Test Acc: {}'.format(epoch,  test_acc))
    
    
    
    
    