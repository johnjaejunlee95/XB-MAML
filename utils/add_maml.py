from collections import OrderedDict
from copy import deepcopy

import torch

from model.conv4 import Conv_Model
from model.resnet import ResNet12
from method.gradient_utils import Finetuning
from method.parameter_utils import sample_model
from utils.utils import *


class Add_MAML:
    def __init__(self, network):
        super(Add_MAML, self).__init__()
        self.network = network
        self.loss = torch.nn.CrossEntropyLoss()
        self.finetuning = Finetuning()

    def add_maml(self, args, ensemble_model, moving_avg, meta_generators):
        sample = self.create_sample_model(args)
        model_parameters = self.get_model_parameters(ensemble_model.model)
        args.meta_lr = args.multi_meta_lr
        
        if len(ensemble_model.model) < 2:
            del ensemble_model.model
            
            ensemble_model.optimizers = None
            ensemble_model.model = []
            
            for i in range(2):
                model_n_param = sample_model(args, parameters=model_parameters)
                model_n = self.create_copy_of_sample_model(sample, model_n_param)
                ensemble_model.add_additional_model(args, model_n)  
                
        else:
            model_n_param = sample_model(args, parameters=model_parameters)
            model_n = self.create_copy_of_sample_model(sample, model_n_param)
            ensemble_model.add_additional_model(args, model_n)

        args.num_maml +=1
        moving_avg.values = []
        
        args.avg_dict = {x: Average(x) for x in meta_generators.stat_keys}
        args.num_overfitting = 5
        self.cleanup(model_n)
        
   
    def create_sample_model(self, args):
        feature_size = args.filter_size*5*5
        if args.model == 'conv4':
            return Conv_Model(args.imgc, args.num_ways, args.filter_size, feature_size, drop_p=args.dropout).cuda()
        else:
            return ResNet12(args.num_ways, drop_p=args.dropout).cuda()

    def get_model_parameters(self, network):
        params = OrderedDict()

        for i in range(len(network)):
            model_parameters = OrderedDict(network[i].meta_named_parameters())
            for key, values in model_parameters.items():
                if key not in params.keys():
                    params[key] = values.detach().cpu() / len(network)
                else:
                    params[key] += values.detach().cpu() / len(network)
        return params

    def create_copy_of_sample_model(self, sample, model_n_param):
        model_n = deepcopy(sample)
        model_n.load_state_dict(model_n_param)
        return model_n

    def cleanup(self, model_n):
        del model_n
        torch.cuda.empty_cache()
