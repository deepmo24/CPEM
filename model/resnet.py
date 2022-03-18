import torch
import torch.nn as nn
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, output_strategy='linear', n_id=80, n_bs=46, n_tex=80, n_rot=3, n_light=27, n_tran=3):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.output_strategy = output_strategy

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        _outdim = 512 * block.expansion

        # building regressor
        self.fc_id = nn.Linear(_outdim, n_id)
        self.fc_bs = nn.Linear(_outdim, n_bs)
        self.fc_tex = nn.Linear(_outdim, n_tex)
        self.fc_angles = nn.Linear(_outdim, n_rot)
        self.fc_light = nn.Linear(_outdim, n_light)
        self.fc_tran = nn.Linear(_outdim, n_tran)
        self.Sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_raw=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # flatten
        bs, out_dim = x.size(0), x.size(1)
        x = x.view(bs, out_dim)

        id_coeff = self.fc_id(x)
        tex_coeff = self.fc_tex(x)
        angle_coeff = self.fc_angles(x)
        light_coeff = self.fc_light(x)
        tran_coeff = self.fc_tran(x)
        # expression coefficients
        bs_coeff = self.Sigmoid(self.fc_bs(x))

        outs = [id_coeff, bs_coeff, tex_coeff, angle_coeff, light_coeff, tran_coeff]
        outs = torch.cat(outs, dim=1)

        return outs



def ResNet50_3DMM(n_id=80, n_bs=46, n_tex=80, n_rot=3, n_light=27, n_tran=3):
    resnet50 = models.resnet50(pretrained=True)
    pretrained_dict = resnet50.state_dict()
    # print('pretrained dict:', len(pretrained_dict.keys()))

    resnet50_3dmm = ResNet(Bottleneck, [3, 4, 6, 3], n_id=n_id, n_bs=n_bs, n_tex=n_tex, n_rot=n_rot, n_light=n_light, n_tran=n_tran)
    model_dict = resnet50_3dmm.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # update model_dict
    model_dict.update(pretrained_dict)

    resnet50_3dmm.load_state_dict(model_dict)

    return resnet50_3dmm