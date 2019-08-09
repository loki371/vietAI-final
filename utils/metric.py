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
    gt_np = y_true.to("cpu").detach().numpy()
    pred_np = (y_pred.to("cpu").detach().numpy() > 0.5) * 1.0
    assert gt_np.shape == pred_np.shape
    for i in range(gt_np.shape[1]):
        f1_out.append(f1_score(gt_np[:,i], pred_np[:,i]))
    return sum(f1_out)/len(f1_out)
