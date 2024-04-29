import torch

def dice_coefficient(predicted:torch.Tensor, target:torch.Tensor, device, labels=[0, 1, 2, 3]):
    pred = torch.argmax(predicted, 1)
    label_channels = []
    for label in labels:
        curr_ch = torch.zeros(pred.shape, dtype=torch.float32)
        curr_ch[pred == label] = 1
        label_channels.append(curr_ch)
    pred = torch.stack(label_channels, 1).to(device)

    intersection = torch.sum(pred * target, (-2, -1))
    total_area = torch.sum(pred, (-2, -1)) + torch.sum(target, (-2, -1))

    eps = 1e-10
    return 2 * (intersection + eps) / (total_area + eps)

def jaccard_index(predicted:torch.Tensor, target:torch.Tensor, device, labels=[0, 1, 2, 3]):
    pred = torch.argmax(predicted, 1)
    label_channels = []
    for label in labels:
        curr_ch = torch.zeros(pred.shape, dtype=torch.float32)
        curr_ch[pred == label] = 1
        label_channels.append(curr_ch)
    pred = torch.stack(label_channels, 1).to(device)

    intersection = torch.sum(pred * target, (-2, -1))
    union = torch.sum(pred, (-2, -1)) + torch.sum(target, (-2, -1)) - intersection

    eps = 1e-10
    return (intersection + eps) / (union + eps)