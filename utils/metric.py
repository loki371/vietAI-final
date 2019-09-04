from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

def multi_class_F1(y_pred, y_true):
    """
    Calculate F1 score for each class
    Parameter
        y_true  -   torch.tensor()
            Groundtruth
        y_pred  -   torch.tensor()
            Prediction
    -------------------------
    Return
        list
            F1 score for each class
    """
    f1_out = []
    gt_np = y_true
    pred_np = (y_pred > 0.5) * 1.0
    assert gt_np.shape == pred_np.shape
    for i in range(gt_np.shape[1]):
        f1_out.append(f1_score(gt_np[:,i], pred_np[:,i]))
    return sum(f1_out)/len(f1_out)

def multi_label_auroc(y_pred, y_gt):
    """ Calculate AUROC for each class

    Parameters
    ----------
    y_gt: torch.Tensor
        groundtruth
    y_pred: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    auroc = []
    # gt_np = y_gt.to("cpu").numpy()
    # gt_np = np.array(gt_np, dtype=np.int32)
    pred_np = (y_pred>0.5)*1.0
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return sum(auroc) / len(auroc)
