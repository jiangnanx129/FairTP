import torch

def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mpe(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse

'''
masked_mape
输入的 null_val 是 0，对 masked_mape 的计算会有以下影响：

将 null_val 设置为 0 时，首先会计算一个布尔类型的 mask，用于表示哪些位置的 labels 值不是缺失值（即非零）。
在计算 mask 值之后，会将其转换为浮点型，并除以 mask 的平均值，目的是进行归一化操作。
接下来，通过使用 torch.where 函数，将 mask 中的所有 NaN 值替换为零。
在计算 loss 时，会计算预测值 preds 与实际值 labels 之间的绝对误差，并除以 labels 的值。这是因为 MAPE（Mean Absolute Percentage Error，平均绝对百分比误差）的定义需要除以真实值，以便计算相对误差。
然后，将 loss 乘以 mask，这样可以将那些缺失值对应的位置的误差置零，保留有效的误差项。
再次使用 torch.where 函数，将 loss 中的所有 NaN 值替换为零。
最后，返回误差 loss 的平均值，即计算有效误差项的平均值。
总结起来，当 null_val 是 0 时，masked_mape 函数会将缺失值对应位置的误差置零，并计算有效误差项的平均值。这样做的目的是消除缺失值对 MAPE 的影响，使得评估结果更加准确。
'''