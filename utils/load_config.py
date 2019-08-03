
from models import VGG16, ResNet50
from torch.optim import Adam, SGD
from torch.nn import BCELoss



def _choose_model(dict_config):
    support_list = ('vgg16', 'resnet50')
    if dict_config['model_name'] not in support_list:
        print(f'Not supported {dict_config["model_name"]}')
        print(f'Only support {support_list}')
        exit()

    if dict_config['model_name'] is 'vgg16':
        dict_config['model'] = VGG16(len(dict_config['class_list']), pretrain=dict_config['pretrain'])
    elif dict_config['model_name'] is 'resnet50':
        dict_config['model'] = ResNet50(len(dict_config['class_list']), pretrain=dict_config['pretrain'])



    return dict_config

def _choose_optim(dict_config):
    support_list = ('Adam', 'SGD')

    if dict_config['optim_name'] not in support_list:
        print(f'Not supported {dict_config["model_name"]}')
        print(f'Only support {support_list}')
        exit()


    if dict_config['optim_name'] is 'Adam':
        dict_config['optimizer'] = Adam(dict_config['model'].parameters(), dict_config['lr'])
    elif dict_config['optim_name'] is 'SGD':
        dict_config['optimizer'] = SGD(dict_config['model'].parameters(), dict_config['lr'])

    return dict_config


def _choose_criterion(dict_config):
    print('Only support BCE Loss function now')
    
    dict_config['criterion'] = BCELoss(weight=dict_config['class_weight'])
    return dict_config




