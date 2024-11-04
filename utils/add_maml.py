# from collections import OrderedDict
# from copy import deepcopy

# import torch

# from model.conv4 import Conv_Model
# from model.resnet import ResNet12
# from method.gradient_utils import Finetuning
# from method.parameter_utils import sample_model
# from utils.utils import *


# class Add_MAML:
#     def __init__(self, network):
#         super(Add_MAML, self).__init__()
#         self.network = network
#         self.loss = torch.nn.CrossEntropyLoss()
#         self.finetuning = Finetuning()

#     def add_maml(self, args, ensemble_model, moving_avg, meta_generators):
#         sample = self.create_sample_model(args)
#         model_parameters = self.get_model_parameters(ensemble_model.model)
#         args.meta_lr = args.multi_meta_lr
        
#         if len(ensemble_model.model) < 2:
#             del ensemble_model.model
            
#             ensemble_model.optimizers = None
#             ensemble_model.model = []
            
#             for i in range(2):
#                 model_n_param = sample_model(args, parameters=model_parameters)
#                 model_n = self.create_copy_of_sample_model(sample, model_n_param)
#                 ensemble_model.add_additional_model(args, model_n)  
                
#         else:
#             model_n_param = sample_model(args, parameters=model_parameters)
#             model_n = self.create_copy_of_sample_model(sample, model_n_param)
#             ensemble_model.add_additional_model(args, model_n)

#         args.num_maml +=1
#         moving_avg.values = []
        
#         args.avg_dict = {x: Average(x) for x in meta_generators.stat_keys}
#         args.num_overfitting = 5
#         self.cleanup(model_n)
        
   
#     def create_sample_model(self, args):
#         feature_size = args.filter_size*5*5
#         if args.model == 'conv4':
#             return Conv_Model(args.imgc, args.num_ways, args.filter_size, feature_size, drop_p=args.dropout).cuda()
#         else:
#             return ResNet12(args.num_ways, drop_p=args.dropout).cuda()

#     def get_model_parameters(self, network):
#         params = OrderedDict()

#         for i in range(len(network)):
#             model_parameters = OrderedDict(network[i].meta_named_parameters())
#             for key, values in model_parameters.items():
#                 if key not in params.keys():
#                     params[key] = values.detach().cpu() / len(network)
#                 else:
#                     params[key] += values.detach().cpu() / len(network)
#         return params

#     def create_copy_of_sample_model(self, sample, model_n_param):
#         model_n = deepcopy(sample)
#         model_n.load_state_dict(model_n_param)
#         return model_n

#     def cleanup(self, model_n):
#         del model_n
#         torch.cuda.empty_cache()


from collections import OrderedDict
from copy import deepcopy

import torch

from model.conv4 import Conv_Model
from model.resnet import ResNet12
from method.parameter_utils import sample_model
from utils.utils import *


class Add_MAML:
    def __init__(self, network):
        super().__init__()
        self.network = network
        self._model_creators = {
            'conv4': lambda args: Conv_Model(
                args.imgc, 
                args.num_ways, 
                args.filter_size, 
                args.filter_size * 5 * 5, 
                drop_p=args.dropout
            ).cuda(),
            'resnet12': lambda args: ResNet12(
                args.num_ways, 
                drop_p=args.dropout
            ).cuda()
        }

    def add_maml(self, args, ensemble_model, moving_avg, meta_generators):
        """Add MAML models to the ensemble with improved handling."""
        sample = self.create_sample_model(args)
        model_parameters = self.get_model_parameters(ensemble_model.model)
        args.meta_lr = args.multi_meta_lr
        
        # Reset ensemble if less than 2 models
        if len(ensemble_model.model) < 2:
            ensemble_model.reset()
            num_models_to_add = 2
        else:
            num_models_to_add = 1
            
        # Add models in batch
        for _ in range(num_models_to_add):
            model_n = self._create_and_add_model(args, sample, model_parameters)
            ensemble_model.add_additional_model(args, model_n)
            
        # Update args and reset moving average
        self._update_args(args, meta_generators)
        moving_avg.values.clear()
        
        # Cleanup
        del sample
        torch.cuda.empty_cache()

    def _create_and_add_model(self, args, sample, model_parameters):
        """Helper method to create and prepare a new model."""
        model_n_param = sample_model(args, parameters=model_parameters)
        model_n = self.create_copy_of_sample_model(sample, model_n_param)
        return model_n

    def _update_args(self, args, meta_generators):
        """Update arguments after adding models."""
        args.num_maml += 1
        args.avg_dict = {x: Average(x) for x in meta_generators.stat_keys}
        args.num_overfitting = 5

    def create_sample_model(self, args):
        """Create a sample model based on architecture type."""
        creator = self._model_creators.get(args.model.lower())
        if not creator:
            raise ValueError(f"Unsupported model type: {args.model}")
        return creator(args)

    def get_model_parameters(self, model_list):
        """Get averaged model parameters with improved efficiency."""
        if not model_list:
            return OrderedDict()
            
        # Initialize with first model's parameters
        params = OrderedDict(model_list[0].meta_named_parameters())
        params = {k: v.detach().cpu() for k, v in params.items()}
        
        # Add remaining models' parameters
        for model in model_list[1:]:
            model_params = OrderedDict(model.meta_named_parameters())
            for key, values in model_params.items():
                params[key] += values.detach().cpu()
        
        # Average the parameters
        num_models = len(model_list)
        for key in params:
            params[key] /= num_models
            
        return params

    @staticmethod
    def create_copy_of_sample_model(sample, model_n_param):
        """Create a copy of the sample model with given parameters."""
        model_n = deepcopy(sample)
        model_n.load_state_dict(model_n_param)
        return model_n