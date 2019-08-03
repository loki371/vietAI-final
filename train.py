from utils import CheXpert_Dataset

from torch.utils.data import DataLoader


dict_config = (
    'train_folder',
    'train_csv',
    
    'handle_uncertain',
    'class_list',
    'img_size',
    'greyscale',

    'val_folder',
    'val_csv',

    'shuffle',
    'batch_size',

    'num_workers'


)



if __name__ == '__main__':
    dict_config = {}


    transform_augment = None


    trainDataset = CheXpert_Dataset(dict_config['train_folder'], dict_config['train_csv'], mode='train', greyscale=dict_config['greyscale']
                                    handle_uncertain=dict_config['handle_uncertain'], transform=transform_augment, 
                                    class_list=dict_config['class_list'], size=dict_config['img_size'])

    trainDataloader = DataLoader(trainDataset, batch_size=dict_config['batch_size'], shuffle=dict_config['shuffle'],
                                 num_workers=dict_config['num_workers'])

    valDataset = CheXpert_Dataset(dict_config['val_folder'], dict_config['val_csv'], mode='val', greyscale=dict_config['greyscale'],
                                  handle_uncertain=dict_config['handle_uncertain'], transform=None, 
                                  class_list=dict_config['class_list'], size=dict_config['img_size'])

    valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False, num_workers=dict_config['num_workers'])



