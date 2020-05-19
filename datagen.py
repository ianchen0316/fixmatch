""" Data Pipeline Before labeled/unlabeled/validation/testing dataloaders are generated """

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_raw_dataset(dataset, root, n_labeled):
     
    """
    A flexible wrapper for generating labeled/unlabeled/validation/testing datasets
    
    Args:
        - root: root path of dataset
        - n_label: number of total labeled data
        - train_transform: transformation pipeline of training data
        - eval_transform: evaluation pipeline of evaluation data
    Returns:
        - labeled_dataset
        - unlabeled_dataset
        - val_dataset
        - test_dataset
    
    
    """
    
    if dataset == 'cifar-10':
        train_base = datasets.CIFAR10(root, train=True, download=True)
        test_base = datasets.CIFAR10(root, train=False, download=True)
    
    # Split original training dataset into labeled/unlabeled/validation indices
    labeled_ind, unlabeled_ind, val_ind = semi_split(train_base.targets, n_labeled, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Create labeled/unlabeled/val/test dataset (before transformation)
    labeled_dataset = (train_base.data[labeled_ind], np.array(train_base.targets)[labeled_ind])
    unlabeled_dataset = train_base.data[unlabeled_ind]
    val_dataset = (train_base.data[val_ind], np.array(train_base.targets)[val_ind])
    test_dataset = (test_base.data, np.array(test_base.targets))
    
    print('# Labeled: {} | # Unlabeled: {}'.format(len(labeled_dataset[0]), len(unlabeled_dataset)))
    
    return labeled_dataset, unlabeled_dataset, val_dataset, test_dataset
    
    
def semi_split(labels, num_label, class_label_ratio):
    
    """ 
    Return the indices to split the dataset into labeled/unlabeled/validation set according to the distribution of class 
    
    Args:
        - labels: original labels of the dataset
        - label_per_class: number of labeled sample per class 
    Returns:
        - labeled_ind: indices of labeled samples
        - unlabeled_ind: indices of unlabeled samples
        - val_ind: indices of validation samples
    
    """
    
    labels = np.array(labels)
    labeled_ind = []
    unlabeled_ind = []
    val_ind = []
    
    # Iterate through each class
    num_classes = len(class_label_ratio)
    
    for i in range(num_classes):
        class_ind = np.where(labels == i)[0]
        np.random.shuffle(class_ind)
        per_class_num = int(num_label*class_label_ratio[i]) 
        labeled_ind.extend(class_ind[:per_class_num])
        unlabeled_ind.extend(class_ind[per_class_num:])
        val_ind.extend(class_ind[-500:])
    
    np.random.shuffle(labeled_ind)
    np.random.shuffle(unlabeled_ind)
    np.random.shuffle(val_ind)
    
    return labeled_ind, unlabeled_ind, val_ind
    

def get_transformed_dataset(labeled_dataset, unlabeled_dataset, val_dataset, test_dataset, weak_transform, strong_transform, eval_transform):
    
    """ Turn raw data into Dataset object with transformations applied.  """
    
    labeled = LabelTransformed(labeled_dataset, weak_transform)
    unlabeled = UnlabelTransformed(unlabeled_dataset, weak_transform, strong_transform)
    valid = ValTransformed(val_dataset, eval_transform)
    test = TestTransformed(test_dataset, eval_transform)
    
    return labeled, unlabeled, valid, test 


def get_dataloader(labeled, unlabeled, valid, test, batch_size, mu):

    
    labeled_iterator = DataLoader(labeled, batch_size=batch_size, shuffle=True)
    unlabeled_iterator = DataLoader(unlabeled, batch_size=mu*batch_size, shuffle=True)
    val_iterator = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_iterator = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    return labeled_iterator, unlabeled_iterator, val_iterator, test_iterator

    
    
class LabelTransformed(Dataset):
    
    def __init__(self, labeled_dataset, weak_transform):
        
        self.labeled_dataset = labeled_dataset[0]
        self.labeled_target = labeled_dataset[1]
        self.weak_transform = weak_transform
    
    def __len__(self):
        
        return len(self.labeled_dataset)
    
    def __getitem__(self, ind):
    
        (img, label) = self.labeled_dataset[ind], self.labeled_target[ind]
        img_weak = self.weak_transform(img)
        
        return img_weak, label

    
class UnlabelTransformed(Dataset):
    
    def __init__(self, unlabeled_dataset, weak_transform, strong_transform):
        
        self.unlabeled_dataset = unlabeled_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        
        return len(self.unlabeled_dataset)
    
    def __getitem__(self, ind):
    
        img = self.unlabeled_dataset[ind]
        img_weak = self.weak_transform(img)
        img_strong = self.strong_transform(img)
        
        return img_weak, img_strong
    

class ValTransformed(Dataset):
    
    def __init__(self, val_dataset, eval_transform):
        
        self.val_dataset = val_dataset[0]
        self.val_target = val_dataset[1]
        self.eval_transform = eval_transform
    
    def __len__(self):
        
        return len(self.val_dataset)
    
    def __getitem__(self, ind):
    
        (img, label) = self.val_dataset[ind], self.val_target[ind]
        img = self.eval_transform(img)
        
        return img, label

    
class TestTransformed(Dataset):
    
    def __init__(self, test_dataset, eval_transform):
        
        self.test_dataset = test_dataset[0]
        self.test_target = test_dataset[1]
        self.eval_transform = eval_transform
    
    def __len__(self):
        
        return len(self.test_dataset)
    
    def __getitem__(self, ind):
    
        (img, label) = self.test_dataset[ind], self.test_target[ind]
        img = self.eval_transform(img)
        
        return img, label
        
