import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from utils import load_config, _choose_model
from utils import CheXpert_Dataset
import argparse

def process(model, test_dataloader, class_list, device='cuda', output_csv='submission.csv'):
    out_pred = torch.FloatTensor().to(device)
    with torch.no_grad(): # Turn off gradient
        # For each batch
        for step, (images, _) in tqdm(enumerate(test_dataloader)):
            # Move images, labels to device (GPU)
            images = images.to(device)

            # Feed forward the model
            ps = model(images)
            out_pred = torch.cat((out_pred, ps), dim=0)

    # TO FILE
    test_df = pd.read_csv("./csv/sample_submission.csv")
    test_df.head()
    label_list = list(class_list)
    for col in test_df.columns[1:]:
        test_df[col] = out_pred[:, label_list.index(col)].cpu().numpy()

    test_df.to_csv(output_csv, index=False)
    test_df.head()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', help='Config inference path')
    args = parser.parse_args()

    print('############ LOAD CONFIG FILE ###############')
    dict_config = load_config(args.config_path)
    ## Load model
    dict_config = _choose_model(dict_config)

    print('############ LOAD TEST DATA ##############')
    test_dataset = CheXpert_Dataset(dict_config['test_folder'], dict_config['test_csv'],
                        mode='test', class_list=dict_config['class_list'],
                        size=dict_config['img_size'], greyscale=dict_config['greyscale'])

    test_loader = DataLoader(test_dataset, batch_size=dict_config['batch_size'], num_workers=0)
    print("Start infer")
    process(
        dict_config['model'],
        test_loader,
        dict_config['class_list'],
        'cuda',
        dict_config['output_csv']
    )
