""" Data Pipeline Before labeled/unlabeled/validation/testing dataloaders are generated """

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_raw_dataset(dataset, root, n_labeled):
     
    """
    A flexible wrapper for generating labeled/unlabeled/validation/testing datasets
    
    Args:
        - root: root path of cifar10 directory
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
    labeled_ind, unlabeled_ind = semi_split(train_base.targets, int(n_labeled/10))
    
    # Create labeled/unlabeled/val/test dataset (before transformation)
    labeled_dataset = (train_base.data[labeled_ind], np.array(train_base.targets)[labeled_ind])
    unlabeled_dataset = train_base.data[unlabeled_ind]
    # val_dataset = (train_base.data[val_ind], np.array(train_base.targets)[val_ind])
    test_dataset = (test_base.data, np.array(test_base.targets))
      
    #labeled_dataset = Cifar10Labeled(root, labeled_ind, train=True, transform_weak=transform_weak)
    #unlabeled_dataset = Cifar10Unlabeled(root, unlabeled_ind, num_transform=1, train=True, transform_weak=transform_weak, transform_strong=transform_strong)
    #val_dataset = Cifar10Labeled(root, val_ind, transform=eval_transform)
    #test_dataset = Cifar10Labeled(root, train=False, transform=eval_transform)
    
    #print("#Labeled: {}, #Unlabeled: {}, #Val: {}, #Test: {}".format(len(labeled_dataset), len(unlabeled_dataset), len(val_dataset), len(test_dataset)))
    
    return labeled_dataset, unlabeled_dataset, test_dataset
    
    
def semi_split(labels, label_per_class):
    
    """ 
    Return the indices to split the dataset into labeled/unlabeled/validation set according to the distribution of class 
    
    #TODO: enable more flexible per-class customization
    #TODO: modify the number of class from 10 to arbitrary
    
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
    # val_ind = []
    
    # Iterate through each class
    for i in range(10):
        class_ind = np.where(labels == i)[0]
        np.random.shuffle(class_ind)
        labeled_ind.extend(class_ind[:label_per_class])
        unlabeled_ind.extend(class_ind[label_per_class:])
        # val_ind.extend(class_ind[-500:])
    
    np.random.shuffle(labeled_ind)
    np.random.shuffle(unlabeled_ind)
    # np.random.shuffle(val_ind)
    
    return labeled_ind, unlabeled_ind
    

def get_transformed_dataset(labeled_dataset, unlabeled_dataset, test_dataset, weak_transform, strong_transform, eval_transform):
    
    """ Turn raw data into Dataset object with transformations applied.  """
    
    labeled = LabelTransformed(labeled_dataset, weak_transform)
    unlabeled = UnlabelTransformed(unlabeled_dataset, weak_transform, strong_transform)
    # valid = ValTransformed(val_dataset, eval_transform)
    test = TestTransformed(test_dataset, eval_transform)
    
    return labeled, unlabeled, test 
    
    
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
        
