import argparse
import random
import pickle
import logging
import os

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from datagen import BatchScenario, LabelTransformed, UnlabelTransformed, EvalTransformed
from randaugment import RandAugmentMC
from model import ModelSetup
from train import get_cosine_schedule_with_warmup, FixmatchLoss, fixmatch_train
from utils import save_checkpoint, evaluate, calculate_accuracy

torch.backends.cudnn.benchmark=True


if __name__ == '__main__':
    
    
    # Parse Hyperparameters:
    parser = argparse.ArgumentParser(description="Fixmatch Hyperparameters")
    
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset for training/evaluating')
    parser.add_argument('--path', type=str, default='./', help='path of the dataset')
    parser.add_argument('--n_labeled', type=int, default=4000, help='number of total labeled data in training')
    
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--model_name', type=str, default='WideResnet', help='backbone model for classification')
    
    
    parser.add_argument('--epochs', type=int, default=1024, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch of labeled data')
    parser.add_argument('--num_iters', type=int, default=1024, help='number of iterations per epoch')
    parser.add_argument('--mu', type=int, default=7, help='ratio of # unlabeled data to # labeled data in training')
    parser.add_argument('--threshold', type=float, default=0.95, help='probability threshold for pseudo label')
    parser.add_argument('--l_u', type=float, default=1.0, help='weight of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='coefficient of L2 regularization loss term')
    parser.add_argument('--seed', type=int, default=42, help='seed for randomization. -1 if no seed')
    
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint. Default set to None if train from scratch')
    parser.add_argument('--model_save_path', type=str, default=None, help='saved path for the model')
    parser.add_argument('--state_path', type=str, default='./states', help='path for states')
    parser.add_argument('--result_path', type=str, default='./results', help='path for results')
    
    args = parser.parse_args()
    
    # Set Loggers and Summary Writers
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO 
    )
    
    os.makedirs(args.state_path, exist_ok=True)
    os.makedirs(args.writer_path, exist_ok=True)
    writer = SummaryWriter(args.writer_path)
 
    # Set Seeds 
    
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        
    # ============ Dataset Setup ============================
    
    config_map = {
    'D_0': [(400, 4600), (400, 4600), (400, 4600), (400, 4600), (400, 4600), (400, 4600), (400, 4600), (400, 4600), (400, 4600), (400, 4600)]  
    }

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
    
    scenario = BatchScenario(args.dataset, args.path, config_map)
    scenario.scenario_generation()
    labeled_dataset, unlabeled_dataset = scenario.get_batch_dataset('D_0')
    test_dataset = scenario.get_test_dataset()
    
    labeled = LabelTransformed(labeled_dataset, weak_transform)
    unlabeled = UnlabelTransformed(unlabeled_dataset, weak_transform, strong_transform)
    test = EvalTransformed(test_dataset, eval_transform)
    
    labeled_iterator = DataLoader(labeled, sampler=RandomSampler(labeled), batch_size=args.batch_size, drop_last=True)
    unlabeled_iterator = DataLoader(unlabeled, sampler=RandomSampler(unlabeled), batch_size=args.mu*args.batch_size, drop_last=True)
    test_iterator = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    
    # ================== Device =====================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================== Log for Settings ===========================
    logger.info(dict(args._get_kwargs()))
    
    # ================= Modeling =====================================
    
    model_setup = ModelSetup(args.n_classes, args.model_name)
    model = model_setup.get_model()
    
    model.to(device)
    
    logger.info("Total Parameters: {}M".format(sum(p.numel()for p in model.parameters()) / 1e6 ))
    
    # ================= Loss function / Optimizer =====================================
    loss_func = FixmatchLoss(args.l_u)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs*args.num_iters)
    
    
    # ================= Training Stage =================================================
    
    start_epoch = 0
    best_acc = 0
    
    if args.resume:
        logger.info(" ================== Resume from checkpoint =========================")
        assert os.path.is_path(args.resume), "Checkpoint directory does not exist!"
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    logger.info(" ===================== Start Training ===================================")
    logger.info(" Num Epochs = {}".format(args.epochs))
    logger.info(" Batch Size = {}".format(args.batch_size))
    logger.info(" Total Optimization Steps: {}".format(args.epochs*args.num_iters))
    
    model.zero_grad()
    
    for epoch in range(start_epoch, args.epochs):
        
        train_loss, train_loss_x, train_loss_u, mask_prob = fixmatch_train(epoch, model, labeled_iterator, unlabeled_iterator, args, loss_func, optimizer, lr_scheduler, device)

        # Will be changed if we use EMA 
        test_model = model
        
        test_acc = evaluate(test_iterator, test_model, epoch)
        
        writer.add_scaler('train/1.train_loss', train_loss, epoch)
        writer.add_scaler('train/2.train_loss_x', train_loss_x, epoch)
        writer.add_scaler('train/3.train_loss_u', train_loss_u, epoch)
        writer.add_scaler('train/4.mask_prob', mask_prob, epoch)
        writer.add_scaler('test/1.test_acc', test_acc, epoch)
        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        state = {'epoch': epoch + 1, 
                 'state_dict': model.state_dict(),
                 'acc': test_acc, 
                 'best_acc': best_acc,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict()}

        save_checkpoint(state, is_best, args.state_path)
        
        logger.info('Best Accuracy: {}'.format(best_acc))
        logger.info('Epoch Test Accuracy: {}'.format(test_acc))
        
    writer.close()
    
    
    
    