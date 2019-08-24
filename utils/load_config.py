import os
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

import sys
sys.path.append(os.path.join(CURRENT_DIR, '../'))

from models import VGG16, ResNet50, EffNet

from torch.optim import Adam, SGD
from torch.nn import BCELoss
import configparser
import ast


from torchvision import transforms
from collections import defaultdict
import torch

ITERAL_LIST = ['class_list', 'translate', 'scale']
FLOAT_LIST = ['lr', 'p_hflip']

def _choose_model(dict_config):
    support_list = ('vgg16', 'resnet50', 'effnet')
    if dict_config['model_name'] not in support_list:
        print(f'Not supported {dict_config["model_name"]}')
        print(f'Only support {support_list}')
        exit()

    if dict_config['model_name'] == 'vgg16':
        print("model = vgg16")
        dict_config['model'] = VGG16(len(dict_config['class_list']), pretrain=dict_config['pretrain'])
    elif dict_config['model_name'] == 'resnet50':
        dict_config['model'] = ResNet50(len(dict_config['class_list']), pretrain=dict_config['pretrain'])
    elif dict_config['model_name'] == 'effnet':
        dict_config['model'] = EffNet(len(dict_config['class_list']), mode=dict_config['eff_mode'])
    
    if dict_config['md_path']:
        state = torch.load(dict_config['md_path'])
        if dict_config['md_key']:
            dict_config['model'].load_state_dict(state[dict_config['md_key']])
        else:
            dict_config['model'].load_state_dict(state)


    return dict_config

def _choose_optim(dict_config):
    support_list = ('Adam', 'SGD')

    if dict_config['optim_name'] not in support_list:
        print(f'Not supported {dict_config["model_name"]}')
        print(f'Only support {support_list}')
        exit()


    if dict_config['optim_name']=='Adam':
        dict_config['optimizer'] = Adam(dict_config['model'].parameters(), dict_config['lr'])
    elif dict_config['optim_name']=='SGD':
        dict_config['optimizer'] = SGD(dict_config['model'].parameters(), dict_config['lr'])

    return dict_config

def _choose_criterion(dict_config, reduction='none'):
    print('Only support BCE Loss function now')
    
    dict_config['criterion'] = BCELoss(weight=dict_config['class_weight'], reduction=reduction)
    return dict_config

def _choose_augmentation(dict_config):
    if not dict_config['use_augmentation']:
        return None
    transform = transforms.Compose([
        transforms.RandomAffine(
            dict_config['rotate'],
            dict_config['translate'],
            dict_config['scale'],
        ),
        transforms.RandomHorizontalFlip(dict_config['p_hflip']),
        transforms.ToTensor()
    ]
    )
    return transform


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def load_configfile(configList, path_configfile):
    # read file .ini
    configList = defaultdict(lambda: None)
    config = configparser.ConfigParser()
    config.read(path_configfile)
    for topic in config:
        for key in config[topic]:
            configList.update({key: config[topic][key]})
    
    # change type
    for element in configList:
        if element in ITERAL_LIST:
            configList[element] = ast.literal_eval(configList[element])
            continue
        if element in FLOAT_LIST:
            configList[element] = float(configList['lr'])
            continue
        if is_int(configList[element]):
            configList[element] = int(configList[element])
            continue
        if (configList[element] == 'True' or configList[element] == 'False'):
            configList[element] = bool(configList[element] == 'True')
            continue
        if (configList[element] == 'None'):
            configList[element] = None
            continue

    configList['translate'] = (float(configList['translate'][0]), float(configList['translate'][1]))
    configList['scale'] = (float(configList['scale'][0]), float(configList['scale'][1]))

    return configList




if __name__ == '__main__':
    # debug
    configList = {}
    configList = load_configfile(configList, './configfiles/template_config.ini')
    for element in configList:
        print(element, "=", configList[element], "- type =", type(configList[element]))


