""" Data Pipeline Before labeled/unlabeled/validation/testing dataloaders are generated """

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class BatchScenario:
    
    def __init__(self, dataset, root, config_map):

        if dataset == 'cifar-10':
        
            train_base = datasets.CIFAR10(root, train=True, download=True)
            
            self._train_data = train_base.data
            self._train_labels = np.array(train_base.targets)
            self._class_indices = [np.where(self._train_labels == i)[0] for i in range(10)]
            
            self.config_map = config_map
            self.total_batch_indices = None 
            
            test_base = datasets.CIFAR10(root, train=False, download=True)
            
            self.test_data = test_base.data
            self.test_labels = np.array(test_base.targets)
            
    def scenario_generation(self):
        
        batch_names = list(self.config_map.keys())
        num_classes = len(self.config_map[batch_names[0]])
        
        total_batches = {'D_0': {'labeled_ind': [], 'unlabeled_ind': []}}
        #total_batches = {}
        #template = {'labeled_ind': [], 'unlabeled_ind': []}
        #for name in batch_names:
        #    total_batches[name] = template
            
        for i in range(num_classes):
            cursor = 0
            np.random.shuffle(self._class_indices[i])
            for name in batch_names:
                n_per_class_labeled = self.config_map[name][i][0]
                total_batches[name]['labeled_ind'].extend(self._class_indices[i][cursor:cursor+n_per_class_labeled])
                cursor += n_per_class_labeled
                # print(self.total_batch_indices['batch_0']['labeled_ind'])
                n_per_class_unlabeled = self.config_map[name][i][1]
                total_batches[name]['unlabeled_ind'].extend(self._class_indices[i][cursor:cursor+n_per_class_unlabeled])
                cursor += n_per_class_unlabeled
        
        self.total_batch_indices = total_batches
        
        
    def get_batch_dataset(self, batch_name):
        
        labeled_ind = self.total_batch_indices[batch_name]['labeled_ind']
        unlabeled_ind = self.total_batch_indices[batch_name]['unlabeled_ind']
        
        labeled_dataset = (self._train_data[labeled_ind], self._train_labels[labeled_ind])
        unlabeled_dataset = self._train_data[unlabeled_ind]
        
        return labeled_dataset, unlabeled_dataset

    def get_test_dataset(self):
        
        test_dataset = (self.test_data, self.test_labels)
        
        return test_dataset
        
        
class LabelTransformed(Dataset):
    
    def __init__(self, labeled_dataset, transform):
        
        self.labeled_dataset = labeled_dataset[0]
        self.labeled_target = labeled_dataset[1]
        self.transform = transform
    
    def __len__(self):
        
        return len(self.labeled_dataset)
    
    def __getitem__(self, ind):
    
        (img, label) = self.labeled_dataset[ind], self.labeled_target[ind]
        img = self.transform(img)
        
        return img, label
       
            
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
            

class EvalTransformed(Dataset):
    
    def __init__(self, eval_dataset, transform):
        
        self.eval_dataset = eval_dataset[0]
        self.eval_target = eval_dataset[1]
        self.transform = transform
    
    def __len__(self):
        
        return len(self.eval_dataset)
    
    def __getitem__(self, ind):
    
        (img, label) = self.eval_dataset[ind], self.eval_target[ind]
        img = self.transform(img)
        
        return img, label
            