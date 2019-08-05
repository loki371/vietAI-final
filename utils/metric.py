from sklearn.metrics import f1_score


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
    gt_np = y_true.to("cpu").numpy()
    pred_np = (y_pred.to("cpu").numpy() > 0.5) * 1.0
    assert gt_np.shape == pred_np.shape
    for i in range(gt_np.shape[1]):
        f1_out.append(f1_score(gt_np[:,i], pred_np[:,i]))
    return f1_out

def multi_label_auroc(y_gt, y_pred):
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
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return auroc