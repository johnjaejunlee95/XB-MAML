from collections import OrderedDict
import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
                                track_running_stats=False)),
        ('relu', nn.ReLU(inplace=True)),
        ('pool', nn.MaxPool2d(2))
    ]))


def conv_drop_block(in_channels, out_channels, drop_p, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
                                track_running_stats=False)),
        ('relu', nn.ReLU(inplace=True)),
        ('dropout', nn.Dropout(drop_p)),
        ('pool', nn.MaxPool2d(2)),
    ]))


class Conv_Model(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64*1*1, drop_p=0.):
        super(Conv_Model, self).__init__()
        
        self.drop_p = drop_p

        kwargs = {}
        if self.drop_p > 0.:
            conv = conv_drop_block
            kwargs['drop_p'] = self.drop_p
            self.drop_classifer = nn.Identity()
        else:
            conv = conv_block
            self.drop_classifer = nn.Identity()

        self.classifier = MetaLinear(feature_size, out_features, bias=True)

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv(in_channels, hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs)),
            ('layer2', conv(hidden_size, hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs)),
            ('layer3', conv(hidden_size, hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs)),
            ('layer4', conv(hidden_size, hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs))
        ]))

    def forward(self, inputs, params=None):
      
        params_feature = self.get_subdict(params, 'features')

        features = self.features(inputs, params=params_feature)
        features = features.view((features.size(0), -1))
       
        params_classifier = self.get_subdict(params, 'classifier')
        
        logits = self.classifier(features, params=params_classifier)
        logits = self.drop_classifer(logits)

        return logits
    