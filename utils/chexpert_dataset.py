import torch
from PIL import Image
from torchvision import transforms

import numpy as np
import pandas as pd




## 0: Negative
## 1: Positive
## -1: Uncertain
CLASS_LIST = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
      'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
      'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
      'Support Devices']

class CheXpert_Dataset():
    def __init__(self, folder, csv_file, mode='train', handle_uncertain='ignore', 
                 transform=None, class_list=CLASS_LIST, greyscale=True,
                size=128):
        ''' Dataset class for CheXpert dataset
        
        Parameters:
            folder: prefix-path for image data
            csv_file: Path to csv file store data
            mode: Train or Test
            
            handle_uncertain: List or string of way to handle uncertain
            transform: Augmentation methods. Remeber add to tensor at last.
            greyscale: bool - Load image as greyscale
            size: int - Image size, size=None for no resize
        '''
        
        
        #### PARSING ARGUMENT
        
        # Convert handle_uncertain to list
        if isinstance(handle_uncertain, str):
            handle_uncertain = [handle_uncertain for x in range(len(class_list))]
        
        if transform is None:
            transform = transforms.ToTensor()
        
        self.transform = transform
        self.mode = mode
        self.greyscale = greyscale
        self.size = size
        #### MAIN IMPLEMENTATION
        data = pd.read_csv(csv_file)
        self.labels = [] #List of tuple (Main label, Extra label)
        self.image_paths = []

        # Load data
        for r in data.iterrows():
            self.image_paths.append(folder + r[1]['Path'])
            if mode is 'train':
                main_label = [r[1][c] for c in class_list]
                # Handle uncertain
                mask = np.zeros(len(CLASS_LIST))
                for i in range(len(main_label)):
                    if main_label[i] == -1:
                        main_label[i], mask[i] = self.handle_uncertain(handle_uncertain[i])
                        
                ex_label = [r[1][c] for c in r[1].keys() if c not in class_list]

                self.labels.append((main_label, ex_label, mask))
            else:
                self.labels.append((None, None, None))
                
            
            
            
    def handle_uncertain(self, way):
        '''Handle uncertain label
        
        way in ['ignore', 'zero', 'one']
        return:
            label
            mask
        '''
        if way is 'ignore':
            return 0, 0
        elif way is 'zero':
            return 0, 1
        elif way is 'one':
            return 1, 1
        else:
            #print('[WARNING] UNDEFINED WAY TO HANDLE UNCERTAIN DATA')
            #print('RETURN LABEL=0, MASK=0')
            return 0, 0
                            
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        '''
        
        Return:
            mode=Train
                x: CPU Tensor image after augmentation - (1, sz, sz)
                y: CPU Tensor Main class labels
                z: Extra class labels
                mask: Mask of label
            mode=Test
                x: CPU Tensor image after augmentation 
                y: None
                z: None
                mask: None
        '''
        
        
        x = Image.open(self.image_paths[index])
        if self.greyscale:
            x = x.convert('L')
        else:
            x = x.convert("RGB")
        
        if self.size is not None:
            x = x.resize((self.size, self.size))
        
        x = self.transform(x)
        
        y = torch.FloatTensor(self.labels[index][0])
        
        masks = torch.FloatTensor(self.labels[index][2])
        return x, y, self.labels[index][1], masks  
    