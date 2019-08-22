import torch


def train_batch(model, optimizer, criterion, train_iter, dict_config, metric_funcs=None):
    ''' Normal training scheme

    input:
        model: Model from _BaseModel
        optimizer
        criterion: Loss function (BCE Loss)
        train_iter
        dict_config
        metric_funcs: Function to calculate metric value that take input: y_pred, y_true


    return:
        loss: Loss value
        metric_values: List of metric value

    '''

    train_iter = train_iter.next()

    imgs, labels, extra_info, mask = train_iter

    imgs = imgs.cuda()
    labels = labels.cuda()

    ### 
    y_pred = model(imgs)
    loss = criterion(y_pred, labels) * mask
    loss = torch.mean(loss)
    
    metric_values = []
    if metric_funcs is not None:
        for metric_func in metric_funcs:
            metric_values.append(metric_func(y_pred, labels))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), metric_values


def train_batch_temporal(model, optimizer, criterion, train_iter, dict_config, metric_funcs=None):
    ''' Training scheme for temporal loss

    input:
        model: Model from _BaseModel
        optimizer
        criterion: Loss function (BCE Loss)
        train_iter
        dict_config
        metric_funcs: Function to calculate metric value that take input: y_pred, y_true


    return:
        loss: Loss value
        metric_values: List of metric value

    '''

    train_iter = train_iter.next()

    imgs1, imgs2, labels, extra_info, mask = train_iter

    ### Forward Feature map 1 
    feature_map1 = model.get_feature_map(imgs1)
    
    fm = feature_map1
    b = fm.shape[0]
    fm = fm.view(b, -1)
    p1 = model.classify(fm)
    
    
    feature_map2 = model.get_feature_map(imgs2)

    fm = feature_map2
    b = fm.shape[0]
    fm = fm.view(b, -1)
    p2 = model.classify(fm)

    

    ## Temporal Loss
    loss = 0
    loss += (feature_map1 - feature_map2) ** 2

    ## BCE Loss
    loss += criterion(p1, labels) * mask
    loss += criterion(p2, labels) * mask


    metric_values = []
    if metric_funcs is not None:
        for metric_func in metric_funcs:
            metric_values.append(metric_func(p1, labels))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), metric_values

