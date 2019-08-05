from utils import CheXpert_Dataset
from utils.metric import multi_class_F1
from utils.evaluate import evaluate

from torch.utils.data import DataLoader
from trainer import train_batch
from utils.load_configfiles import load_configfiles

import torch

dict_config = (
    'train_folder', #
    'train_csv', #
    
    'handle_uncertain',
    'class_list', #
    'img_size',
    'greyscale',

    'val_folder', #
    'val_csv', #

    'shuffle',
    'batch_size',

    'num_workers',
    'nEpochs',

    'train_pfeq',   #Print frequent for train
    'val_pfeq',     #Print frequent for val


    'pretrain'
)



if __name__ == '__main__':
    dict_config = load_configfiles()
    for c in dict_config:
        print("c = ", c)

    transform_augment = None

    print("before bug 1")
    trainDataset = CheXpert_Dataset(dict_config['train_folder'], dict_config['train_csv'], mode='train', greyscale=dict_config['greyscale'],
                                    handle_uncertain=dict_config['handle_uncertain'], transform=transform_augment, 
                                    class_list=dict_config['class_list'], size=dict_config['img_size'])
    print("after bug 1")
    trainDataloader = DataLoader(trainDataset, batch_size=dict_config['batch_size'], shuffle=dict_config['shuffle'],
                                 num_workers=dict_config['num_workers'])

    valDataset = CheXpert_Dataset(dict_config['val_folder'], dict_config['val_csv'], mode='val', greyscale=dict_config['greyscale'],
                                  handle_uncertain=dict_config['handle_uncertain'], transform=None, 
                                  class_list=dict_config['class_list'], size=dict_config['img_size'])

    valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False, num_workers=dict_config['num_workers'])



    model = dict_config['model']
    optimizer = dict_config['optimizer']
    criterion = dict_config['criterion']

    best_F1_score = 0
    best_val_loss = 10000
    for e in range(dict_config['nEpochs']):
        train_iter = iter(trainDataloader)
        eLoss = 0
        emetric_value = {'F1':0}
        for b in range(len(trainDataloader)):
            bLoss, bMetrics = train_batch(model, optimizer, criterion, train_iter, dict_config, metric_funcs=[multi_class_F1])

            eLoss += bLoss 

            emetric_value['F1'] += bMetrics[0]

            if b % dict_config['train_pfeq'] == 1:
                print(f'Epoch {e} - [{b} / {len(trainDataloader)}]:\nLoss: {eLoss / b}')
                print(f"F1: {emetric_value['F1'] / b}")
        
        if e % dict_config['save_checkpoint_feq'] == 0:
            torch.save(model.state_dict(), dict_config['checkpoint_path'])
        
        val_loss, val_F1  = evaluate(model, valDataloader, criterion, None, dict_config['val_pfeq'])



        if best_F1_score < val_F1:            
            print(f'Updated best F1 from {best_F1_score} to {val_F1}')
            torch.save(model.state_dict(), dict_config['bestF1_path'])
            best_F1_score = val_F1

        if best_val_loss > val_loss:
            print(f"Update best loss from {best_val_loss} to {val_loss}")
            torch.save(model.state_dict(), dict_config['best_valLoss_path'])
            best_val_loss = val_loss