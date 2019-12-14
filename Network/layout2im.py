import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineModule(nn.Module):
    def __init__(self, layout_size, input_size, output_size, num_convs=2):
        super(RefineModule, self).__init__()
        layers = []
        sizes = [layout_size+input_size, output_size, output_size]
        for i in range(num_convs):
            c = nn.Conv2d(sizes[i], sizes[i+1], kernel_size=3, padding=1)
            nn.init.kaiming_normal_(c.weight)
            layers.append(c)
            layers.append(nn.BatchNorm2d(output_size))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.network = nn.Sequential(*layers)

    def forward(self, layout, features):
        _, _, H0, _ = features.size()
        factor = 64 // H0
        layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
        network_input = torch.cat([layout, features], dim=1)
        return self.network(network_input)


class RefineNetwork(nn.Module):
    def __init__(self, num_convs=2, num_layers=5):
        super(RefineNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(RefineModule(128, 32, 1024))
        self.layers.append(RefineModule(128, 1024, 512))
        self.layers.append(RefineModule(128, 512, 256))
        self.layers.append(RefineModule(128, 256, 128))
        self.layers.append(RefineModule(128, 128, 64))
        self.num_layers = num_layers

        output_layers = []
        sizes, kernels, paddings = [64, 64, 3], [3, 1], [1, 0]
        for i in range(num_convs):
            c = nn.Conv2d(sizes[i], sizes[i+1],
                          kernel_size=kernels[i], padding=paddings[i])
            nn.init.kaiming_normal_(c.weight)
            output_layers.append(c)
            if i < (num_convs-1):
                output_layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.output = nn.Sequential(*output_layers)

    def forward(self, layout):
        N, _, H, W = layout.size()
        features = torch.randn(
            (N, 32, 2, 2), dtype=layout.dtype, device=layout.device)
        for layer in self.layers:
            features = F.interpolate(features, scale_factor=2, mode='nearest')
            features = layer(layout, features)
        images = self.output(features)
        return images
