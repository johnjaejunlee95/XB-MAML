from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

"""
ResNet
from https://github.com/kjunelee/MetaOptNet
This ResNet network was designed following the practice of the following papers:
TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
"""


def get_subdict(adict, name):
    if adict is None:
        return adict
    tmp = {k[len(name) + 1:]:adict[k] for k in adict if name in k}
    return tmp


class DropBlock(MetaModule):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            # print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            # print (block_mask.size())
            # print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                downsample=None, drop_rate=0.0, drop_block=False,
                block_size=1, max_padding=0, track_running_stats=False):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu3 = nn.LeakyReLU()

        self.maxpool = nn.MaxPool2d(stride=stride, kernel_size=[stride,stride],
                                                            padding=max_padding)
        self.max_pool = True if stride != max_padding else False
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.dropout = nn.Dropout(self.drop_rate)
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        residual = x
        out = self.conv1(x, params=get_subdict(params, 'conv1'))
        out = self.bn1(out, params=get_subdict(params, 'bn1'))
        out = self.relu1(out)

        out = self.conv2(out, params=get_subdict(params, 'conv2'))
        out = self.bn2(out, params=get_subdict(params, 'bn2'))
        out = self.relu2(out)

        out = self.conv3(out, params=get_subdict(params, 'conv3'))
        out = self.bn3(out, params=get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual = self.downsample(x, params=get_subdict(params, 'downsample'))
        out += residual
        out = self.relu3(out)

        if self.drop_rate > 0:
            out = self.dropout(out)

        if self.max_pool:
            out = self.maxpool(out)

        return out


def mlp(in_dim, out_dim):
    return MetaSequential(OrderedDict([
        ('linear1', MetaLinear(in_dim, out_dim, bias=True)),
        ('relu', nn.ReLU()),
        ('linear2', MetaLinear(out_dim, out_dim, bias=True)),
    ]))


def mlp_drop(in_dim, out_dim, drop_p):
    return MetaSequential(OrderedDict([
        ('linear1', MetaLinear(in_dim, out_dim, bias=True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(drop_p)),
        ('linear2', MetaLinear(out_dim, out_dim, bias=True)),
    ]))


class ResNet(MetaModule):
    def __init__(self, blocks, avg_pool=True, drop_rate=0.0, dropblock_size=5,
                 out_features=5, wh_size=1, inductive_bn=False):
        self.inplanes = 3
        super(ResNet, self).__init__()
        self.anil = False

        self.inductive_bn = inductive_bn
        self.layer1 = self._make_layer(blocks[0], 64, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer2 = self._make_layer(blocks[1], 128, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer3 = self._make_layer(blocks[2], 256, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(blocks[3], 512, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)

        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop_rate = drop_rate
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=self.drop_rate, inplace=False)
        self.classifier = MetaLinear(512 * wh_size * wh_size, out_features)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, MetaLinear):
                nn.init.xavier_uniform_(m.weight)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0,
                    drop_block=False, block_size=1, max_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=1, bias=False),
                MetaBatchNorm2d(planes * block.expansion,
                                track_running_stats=self.inductive_bn),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                    downsample, drop_rate, drop_block, block_size, max_padding, track_running_stats=self.inductive_bn))
        self.inplanes = planes * block.expansion
        return MetaSequential(*layers)

    def forward(self, x, params=None):
        if self.anil:
            params_feature = [None for _ in range(4)]
        else:
            params_feature = [get_subdict(params, f'layer{i+1}') for i in range(4)]

        x = self.layer1(x, params=params_feature[0])
        x = self.layer2(x, params=params_feature[1])
        x = self.layer3(x, params=params_feature[2])
        x = self.layer4(x, params=params_feature[3])
        if self.keep_avg_pool:
            x = self.avgpool(x)
        features = x.view((x.size(0), -1))
        features = self.dropout(features)
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))

        return logits


def ResNet12(num_classes, drop_p=0., inductive_bn=False):
    blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
    return ResNet(blocks, drop_rate=drop_p, out_features=num_classes, inductive_bn=inductive_bn)


# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class ResNet12(nn.Module):
    
#     def init_weights(self):
#         torch.manual_seed(43)
#         torch.cuda.manual_seed(542)
#         torch.cuda.manual_seed_all(117)
#         print('init weights')
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight)
                
#     def __init__(self, in_ch, n_ways):
#         super(ResNet12, self).__init__()
        
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(num_features=64, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(num_features=64, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(num_features=64, eps=2e-05))
#         self.conv1_r = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
#                                      nn.BatchNorm2d(num_features=64))
#         self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True))

#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(128, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(128, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(128, eps=2e-05))
#         self.conv2_r = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
#                                      nn.BatchNorm2d(128, eps=2e-05))
#         self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True))

#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(256, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(256, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(256, eps=2e-05))
#         self.conv3_r = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
#                                      nn.BatchNorm2d(256, eps=2e-05))
#         self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True))

#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(512, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(512, eps=2e-05),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
#                                    nn.BatchNorm2d(512, eps=2e-05))
#         self.conv4_r = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
#                                      nn.BatchNorm2d(512, eps=2e-05))
#         self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, ceil_mode=True),
#                                    nn.AvgPool2d(kernel_size=6))
        
#         self.convblock = nn.Sequential(OrderedDict([
#             ('conv1', self.conv1),
#             ('conv1_r', self.conv1_r),
#             ('conv2', self.conv2),
#             ('conv2_r', self.conv2_r),
#             ('conv3', self.conv3),
#             ('conv3_r', self.conv3_r),
#             ('conv4', self.conv4),
#             ('conv4_r', self.conv4_r),
#             ('pool4', self.pool4),
#         ]))
#         self.add_module('fc', nn.Linear(512, n_ways))
#         self.init_weights()
        
#     def conv(self, input, weight, bias, stride=1, padding=1):        
#         conv = F.conv2d(input, weight, bias, stride, padding)
#         return conv
        
#     def batchnorm(self, input, weight, bias):
#         running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).cuda()
#         running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).cuda()
#         batchnorm = F.batch_norm(input, running_mean, running_var, weight, bias, training=True, eps=2e-5)
#         return batchnorm
    
#     def relu(self, input):
#         relu = F.relu(input, inplace=True)
#         return relu
        
#     def maxpool(self, input, kernel_size=2, stride=2, padding=0 ):
#         maxpool = F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding)        
#         return maxpool
        
#     def forward(self, x, weights=None):
#         if weights == None:
#             x = self.convblock(x)
#             x = x.view(x.size(0), -1)
#             x = self.fc(x)
#         else:
#             weights_idx = list(weights.keys())
#             x1 = x
#             x1_1 = x
#             x1 = self.conv(x1, weights[weights_idx[0]], weights[weights_idx[1]])
#             x1 = self.batchnorm(x1, weights[weights_idx[2]], weights[weights_idx[3]])
#             x1 = self.relu(x1)
#             x1 = self.conv(x1, weights[weights_idx[4]], weights[weights_idx[5]])
#             x1 = self.batchnorm(x1, weights[weights_idx[6]], weights[weights_idx[7]])
#             x1 = self.relu(x1)
#             x1 = self.conv(x1, weights[weights_idx[8]], weights[weights_idx[9]])
#             x1 = self.batchnorm(x1, weights[weights_idx[10]], weights[weights_idx[11]])
#             x1_1 = self.conv(x1_1, weights[weights_idx[12]], weights[weights_idx[13]])
#             x1_1 = self.batchnorm(x1_1, weights[weights_idx[14]], weights[weights_idx[15]])
           
#             out1 = self.pool1(F.relu(x1 + x1_1))
#             out1_1 = out1.clone()

#             out1 = self.conv(out1, weights[weights_idx[16]], weights[weights_idx[17]])
#             out1 = self.batchnorm(out1, weights[weights_idx[18]], weights[weights_idx[19]])
#             out1 = self.relu(out1)
#             out1 = self.conv(out1, weights[weights_idx[20]], weights[weights_idx[21]])
#             out1 = self.batchnorm(out1, weights[weights_idx[22]], weights[weights_idx[23]])
#             out1 = self.relu(out1)
#             out1 = self.conv(out1, weights[weights_idx[24]], weights[weights_idx[25]])
#             out1 = self.batchnorm(out1, weights[weights_idx[26]], weights[weights_idx[27]])
            
#             out1_1 = self.conv(out1_1, weights[weights_idx[28]], weights[weights_idx[29]])
#             out1_1 = self.batchnorm(out1_1, weights[weights_idx[30]], weights[weights_idx[31]])
            
#             out2 = self.pool2(F.relu(out1 + out1_1))
#             out2_1 = out2.clone()

#             out2 = self.conv(out2, weights[weights_idx[32]], weights[weights_idx[33]])
#             out2 = self.batchnorm(out2, weights[weights_idx[34]], weights[weights_idx[35]])
#             out2 = self.relu(out2)
#             out2 = self.conv(out2, weights[weights_idx[36]], weights[weights_idx[37]])
#             out2 = self.batchnorm(out2, weights[weights_idx[38]], weights[weights_idx[39]])
#             out2 = self.relu(out2)
#             out2 = self.conv(out2, weights[weights_idx[40]], weights[weights_idx[41]])
#             out2 = self.batchnorm(out2, weights[weights_idx[42]], weights[weights_idx[43]])
#             out2_1 = self.conv(out2_1, weights[weights_idx[44]], weights[weights_idx[45]])
#             out2_1 = self.batchnorm(out2_1, weights[weights_idx[46]], weights[weights_idx[47]])
            
#             out3 = self.pool3(F.relu(out2 + out2_1))
#             out3_1 = out3.clone()
            
#             out3 = self.conv(out3, weights[weights_idx[48]], weights[weights_idx[49]])
#             out3 = self.batchnorm(out3, weights[weights_idx[50]], weights[weights_idx[51]])
#             out3 = self.relu(out3)
#             out3 = self.conv(out3, weights[weights_idx[52]], weights[weights_idx[53]])
#             out3 = self.batchnorm(out3, weights[weights_idx[54]], weights[weights_idx[55]])
#             out3 = self.relu(out3)
#             out3 = self.conv(out3, weights[weights_idx[56]], weights[weights_idx[57]])
#             out3 = self.batchnorm(out3, weights[weights_idx[58]], weights[weights_idx[59]])
#             out3_1 = self.conv(out3_1, weights[weights_idx[60]], weights[weights_idx[61]])
#             out3_1 = self.batchnorm(out3_1, weights[weights_idx[62]], weights[weights_idx[63]])
            
#             out4 = self.pool4(F.relu(out3 + out3_1))
#             h_t = out4.view(x.shape[0], -1)
#             output = F.linear(h_t, weights[weights_idx[64]],  weights[weights_idx[65]])
#         return h_t, output