import torch
from torch import nn
from torch.autograd import Function
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import models


# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## AlexNet #############################

class AlexNetBase(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.output_col = 4096

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


########## Swin #############################

class SwinTiny(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinTiny, self).__init__()
        self.output_type = 'pooler_output'
        if pretrained:
            import transformers
            swin_module = transformers.SwinForImageClassification
            self.model_swin = swin_module.from_pretrained("microsoft/swin-tiny-patch4-window7-224").swin
        else:
            raise NotImplementedError
        self.output_col = 768

    def forward(self, x):
        x = self.model_swin(x)
        return x[self.output_type]


########## ResNet  #############################
model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or \
       classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nobn = nobn

    def forward(self, x, source=True):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        print(self.scale)
        return input * self.scale


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.nobn = nobn

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


class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.output_col = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, nobn=False):
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
            layers.append(block(self.inplanes, planes, nobn=nobn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Resnet18(Resnet):
    def __init__(self, num_classes=1000, pretrained=True):
        layers = [2, 2, 2, 2]
        block = BasicBlock
        super().__init__(block, layers, num_classes)

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))


class Resnet34(Resnet):
    def __init__(self, num_classes=1000, pretrained=True):
        layers = [3, 4, 6, 3]
        block = BasicBlock
        super().__init__(block, layers, num_classes)

        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))




# Classifier
class Classifier(nn.Module):
    def __init__(self, num_class, src_class, trg_class, inc=512, temp=0.05, pretrain=False):
        super().__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.src_class = src_class
        self.trg_class = trg_class
        self.num_class = num_class
        self.pretrain = pretrain
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1, domain_type=None, shared_indicator=False):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        if shared_indicator:
            trg_mask = torch.zeros_like(x_out)
            src_mask = torch.zeros_like(x_out)

            trg_mask[:, self.trg_class] = 1
            src_mask[:, self.src_class] = 1

            class_mask = trg_mask.logical_and(src_mask)
            x_out[class_mask == 0] = -100
        else:
            if not self.pretrain:
                class_mask = torch.zeros_like(x_out)
                if domain_type == 'src':
                    class_mask[:, self.src_class] = 1
                elif domain_type == 'trg':
                    class_mask[:, self.trg_class] = 1
                elif domain_type == 'all':
                    class_mask = torch.ones_like(x_out)
                else:
                    raise NotImplementedError
                # set logits of classes not in domain as -100
                x_out[class_mask == 0] = -100
        return x_out

class Predictor(nn.Module):
    def __init__(self, num_class,src_class, trg_class, inc=4096, temp=0.05, hidden=[],
                 normalize=False, cls_bias=True, pretrain=False):
        super(Predictor, self).__init__()
        layer_nodenums = [inc] + hidden + [num_class]
        layers = []
        for i in range(len(layer_nodenums)-1):
            if i==len(layer_nodenums)-2:
                bias = cls_bias
            else:
                bias = True
            layers.append(nn.Linear(layer_nodenums[i], layer_nodenums[i+1], bias=bias))

        self.normalize = normalize
        self.fc = nn.Sequential(*layers)
        self.num_class = num_class
        self.src_class = src_class
        self.trg_class = trg_class
        self.pretrain = pretrain
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1, domain_type=None):
        if reverse:
            x = grad_reverse(x, eta)
        if self.normalize:
            x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        #print(x_out.shape)
        if not self.pretrain:
            class_mask = torch.zeros_like(x_out)
            if domain_type == 'src':
                class_mask[:, self.src_class] = 1
            elif domain_type == 'trg':
                class_mask[:, self.trg_class] = 1
            else:
                raise NotImplementedError
            # set logits of classes not in domain as -100
            x_out[class_mask == 0] = -100
            #assert not torch.any(x_out == -100), "Error: x_out contains -100"
        return x_out