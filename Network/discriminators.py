import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDiscriminator(nn.Module):
    def __init__(self, num_convs=3):
        super(ImageDiscriminator, self).__init__()
        layers = []
        sizes = [3, 64, 128, 256]

        for i in range(num_convs):
            c = nn.Conv2d(sizes[i], sizes[i+1],
                          kernel_size=4, padding=1, stride=2)
            nn.init.kaiming_normal_(c.weight)
            layers.append(c)
            if i < (num_convs-1):
                layers.append(nn.BatchNorm2d(sizes[i+1]))
                layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(sizes[-1], 1, kernel_size=1, stride=1)

    def forward(self, img_preds):
        return self.network(img_preds)


class ObjectDiscriminator(nn.Module):
    def __init__(self, num_objects, num_convs=3):
        super(ObjectDiscriminator, self).__init__()
        self.num_objects = num_objects
        layers = []
        sizes = [3, 64, 128, 256]

        for i in range(num_convs):
            c = nn.Conv2d(sizes[i], sizes[i+1],
                          kernel_size=4, padding=1, stride=2)
            nn.init.kaiming_normal_(c.weight)
            layers.append(c)
            if i < (num_convs-1):
                layers.append(nn.BatchNorm2d(sizes[i+1]))
                layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.network = nn.Sequential(nn.Sequential(
            *layers), GlobalAvgPool(), nn.Linear(sizes[-1], 1024))
        self.real = nn.Linear(1024, 1)
        self.object = nn.Linear(1024, num_objects)

    def forward(self, img_preds, object_indexs, boxes, obj2img, weight=0.1):
        crops = crop_batch(img_preds, boxes, obj2img)
        out = self.network(crops)
        real_scores = self.real(out)
        object_scores = self.object(out)
        aux_loss = F.cross_entropy(object_scores, object_indexs-1)
        return real_scores, weight*aux_loss


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        N, C = x.size(0), x.size(1)
        return x.view(N, C, -1).mean(dim=2)


def crop_batch(img_preds, boxes, obj2img):
    N, C, H, W = img_preds.size()
    B = boxes.size(0)

    images_flat, boxs_flat, all_indexs = [], [], []
    for i in range(N):
        indexs = (obj2img.data == i).nonzero()
        if indexs.dim() == 0:
            continue
        indexs = indexs.view(-1)
        n = indexs.size(0)
        image = img_preds[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        box = boxes[indexs]

        images_flat.append(image)
        boxs_flat.append(box)
        all_indexs.append(indexs)

    images_flat = torch.cat(images_flat, dim=0)
    boxs_flat = torch.cat(boxs_flat, dim=0)
    all_indexs = torch.cat(all_indexs, dim=0)
    crops = crop_box(images_flat, boxs_flat, 32, 32)

    if (all_indexs == torch.arange(0, B).type_as(all_indexs)).all():
        return crops
    return crops[permu(all_indexs)]


def crop_box(images_flat, boxs_flat, HH, WW):
    N = images_flat.size(0)
    boxs_flat = 2 * boxs_flat - 1
    x0, y0 = boxs_flat[:, 0], boxs_flat[:, 1]
    x1, y1 = boxs_flat[:, 2], boxs_flat[:, 3]
    X = linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    grid = torch.stack([X, Y], dim=3)
    return F.grid_sample(images_flat, grid)


def linspace(start, end, steps=10):
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)
    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)
    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)
    out = start_w * start + end_w * end
    return out


def permu(p):
    N = p.size(0)
    eye = torch.arange(0, N).type_as(p)
    pp = (eye[:, None] == p).nonzero()[:, 1]
    return pp
