import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    def __init__(self, input_size=128, hidden_size=512, output_size=128):
        super(GraphConvLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net1 = L2Perceptron(
            3*input_size, hidden_size, 2*hidden_size+output_size)
        self.net2 = L2Perceptron(hidden_size, hidden_size, output_size)

    def forward(self, things, predicates, edges):
        O = things.size(0)
        T = predicates.size(0)
        #D_in = self.input_size
        H = self.hidden_size
        D_out = self.output_size

        s = edges[:, 0].contiguous()
        o = edges[:, 1].contiguous()
        subjects = things[s]
        objects = things[o]

        triples = torch.cat([subjects, predicates, objects], dim=1)
        new_triples = self.net1(triples)
        new_subjects = new_triples[:, :H]
        new_predicates = new_triples[:, H:(H+D_out)]
        new_objects = new_triples[:, (H+D_out):]

        # Average (pooling function h) candidates
        pool_things = torch.zeros(
            O, H, dtype=things.dtype, device=things.device)
        pool_things = pool_things.scatter_add(
            0, s.view(-1, 1).expand_as(new_subjects), new_subjects)
        pool_things = pool_things.scatter_add(
            0, o.view(-1, 1).expand_as(new_objects), new_objects)

        counts = torch.zeros(O, dtype=things.dtype, device=things.device)
        ones = torch.ones(T, dtype=things.dtype, device=things.device)
        counts = counts.scatter_add(0, s, ones)
        counts = counts.scatter_add(0, o, ones)
        counts = counts.clamp(min=1)
        pool_things = pool_things / counts.view(-1, 1)
        new_things = self.net2(pool_things)
        return new_things, new_predicates


class GraphConvNetwork(nn.Module):
    def __init__(self, num_layers=5):
        super(GraphConvNetwork, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(GraphConvLayer())

    def forward(self, things, predicates, edges):
        for i in range(self.num_layers):
            layer = self.layers[i]
            things, predicates = layer(things, predicates, edges)
        return things, predicates


def L2Perceptron(input_size, hidden_size, output_size, num_layers=2):
    layers = []
    sizes = [input_size, hidden_size, output_size]
    for i in range(num_layers):
        fc = nn.Linear(sizes[i], sizes[i+1])
        nn.init.kaiming_normal_(fc.weight)
        layers.append(fc)
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def BoxNetwork():
    return L2Perceptron(128, 512, 4)


def MaskNetwork(mask_size=16, input_dim=128, output_dim=1):
    layers = []
    size = 1
    while size < mask_size:
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.BatchNorm2d(input_dim))
        layers.append(nn.Conv2d(input_dim, input_dim,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        size *= 2
    layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)


def cast(things, boxes, masks, obj2img):
    O, D = things.size()
    M = masks.size(1)
    grid = build_grid(boxes)
    imgs = things.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    samples = F.grid_sample(imgs, grid)
    layout = build_layout(samples, obj2img)
    return layout


def build_grid(boxes, H=64, W=64):
    O = boxes.size(0)
    boxes = boxes.view(O, 4, 1, 1)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x1 - x0
    h = y1 - y0
    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)
    X = (X - x0) / w
    Y = (Y - y0) / h
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)
    grid = grid.mul(2).sub(1)
    return grid


def build_layout(samples, obj2img):
    O, D, H, W = samples.size()
    N = obj2img.data.max().item() + 1
    layout = torch.zeros(N, D, H, W, dtype=samples.dtype,
                         device=samples.device)
    indices = obj2img.view(O, 1, 1, 1).expand(O, D, H, W)
    layout = layout.scatter_add(0, indices, samples)
    return layout
