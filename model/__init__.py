import torch
from .conv4 import Conv_Model
from .resnet import ResNet12
from utils.utils import MovingAverage
from method import xb_maml



def model_selection(args, meta_generators, ensemble_model):
    
    if args.model == 'conv4':
        feature_size = args.filter_size*5*5
        model = Conv_Model(args.imgc, args.num_ways, args.filter_size, feature_size, drop_p=args.dropout).cuda()
    elif args.model == 'resnet':
        model = ResNet12(args.num_ways, drop_p=args.dropout).cuda()
    
    moving_avg = MovingAverage(threshold_count=args.threshold)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    maml = xb_maml.XB_MAML(model, ensemble_model, optimizer, meta_generators, moving_avg)
  
    return maml, model