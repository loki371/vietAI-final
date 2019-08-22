import torch
from .metric import multi_class_F1
import json

def evaluate(model, val_dataloader, criterion, log_file=None, print_feq=100):
    model.eval()
    detail_eval = []
    
    total_f1 = 0
    total_loss = 0
    c = 0


    with torch.no_grad():
        for i, (imgs, labels, extra_info, mask) in enumerate(val_dataloader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            
            y_pred = model(imgs)
            for j in range(len(y_pred)):
                detail_eval.append({
                    'y_pred': y_pred[j],
                    'y_label': labels[j],
                    'extra_info': extra_info
                })
        
            val_batch_size = len(y_pred)
            total_f1 = multi_class_F1(y_pred, labels) * val_batch_size
            total_loss = criterion(y_pred, labels) * val_batch_size
            c += val_batch_size

        if i % print_feq == 1:
            print(f'Validation {i} - Loss: {total_loss / c} - F1: {total_f1 / c}')
        
    if log_file is not None:
        json.dump(detail_eval, open(log_file, 'w'))
    

    print(f'END OF VALIDATION:\nLoss:{total_loss / c}\nF1:{total_f1/c}')
    return total_loss / c, total_f1 / c

