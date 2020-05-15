import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from datagen import get_raw_dataset, get_transformed_dataset
from randaugment import RandAugmentMC
from model import VGG, WideResnet
from train import FixmatchLoss, fixmatch_train
from utils import evaluate, calculate_accuracy


if __name__ == '__main__':
    
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
    
    labeled_dataset, unlabeled_dataset, val_dataset, test_dataset = get_raw_dataset('cifar-10', './', 
                                                                                n_labeled=2000)
    
    labeled, unlabeled, valid, test = get_transformed_dataset(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, weak_transform, strong_transform, eval_transform)
    
    labeled_iterator = DataLoader(labeled, batch_size=32, shuffle=True)
    unlabeled_iterator = DataLoader(unlabeled, batch_size=128, shuffle=True)
    val_iterator = DataLoader(valid, batch_size=32, shuffle=False)
    test_iterator = DataLoader(test, batch_size=32, shuffle=False)

    # ================== Device =====================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # ================= Modeling =====================================
    model = WideResnet(n_classes=10)
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
    
    
    
    
    