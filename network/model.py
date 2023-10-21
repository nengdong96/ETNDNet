
import torchvision
import torch.nn as nn
from .gem_pool import GeneralizedMeanPoolingP

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Model(nn.Module):

    def __init__(self):
        super(Model, self,).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool, resnet.layer1,
                                         resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        features_map = self.resnet_conv(x)
        return features_map

class Classifier(nn.Module):
    def __init__(self, pid_num):
        super(Classifier, self, ).__init__()
        self.pid_num = pid_num
        self.GAP = GeneralizedMeanPoolingP()
        self.BN = nn.BatchNorm1d(2048)
        self.BN.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features_map):
        features = self.GAP(features_map)
        bn_features = self.BN(features.squeeze())
        cls_score = self.classifier(bn_features)
        if self.training:
            return cls_score
        else:
            return bn_features


