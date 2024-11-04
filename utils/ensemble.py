import torch
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules import MetaModule
from method.parameter_utils import linear_combination

# class Ensemble_Model(MetaModule):
#     def __init__(self):
#         super(Ensemble_Model, self).__init__()

#         self.optimizers = None
#         self.model = []

#     def add_additional_model(self, args, new_network, status='train'):
       
#         self.model.append(new_network)
            
#         if status == 'train':
#             if self.optimizers is None:
#                 self.optimizers = torch.optim.Adam(new_network.parameters(), lr=args.meta_lr)
#             else:
#                 self.optimizers.add_param_group({'params': new_network.parameters(), 'lr': args.meta_lr})
#         else:
#             pass
            
#     def _compute_final_weights(self, args, param_list, train_losses=None):
#         sigma = F.softmax(-torch.stack(train_losses) / args.temp_scaling, dim=0)
#         final_weights = linear_combination(param_list, sigma=sigma)
#         return sigma, final_weights

#     def outputs(self, args, batch=None):
                
#         train_losses = []
#         param_list = []
#         train_input, train_target = batch
        
#         for i in range(len(self.model)):
#             self.model[i].eval()
#             logits = self.model[i](train_input)
#             train_loss = F.cross_entropy(logits, train_target)
#             self.model[i].zero_grad()
#             train_losses.append(train_loss)
#             param_list.append(OrderedDict(self.model[i].meta_named_parameters()))

#         sigmas, weights = self._compute_final_weights(args, param_list, train_losses)

#         return weights, sigmas


class Ensemble_Model(MetaModule):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.models = torch.nn.ModuleList()  # Using ModuleList for better model management
    
    def add_additional_model(self, args, new_network, status='train'):
        """Add a new model to the ensemble and configure its optimizer if in training mode.
        
        Args:
            args: Configuration arguments
            new_network: Neural network model to add
            status: 'train' or 'eval' mode
        """
        self.models.append(new_network)
        
        if status == 'train':
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(new_network.parameters(), lr=args.meta_lr)
            else:
                self.optimizer.add_param_group({'params': new_network.parameters(), 'lr': args.meta_lr})

    @torch.no_grad()  # Optimization: disable gradient computation for inference
    def _compute_final_weights(self, args, param_list, train_losses):
        """Compute ensemble weights using softmax with temperature scaling.
        
        Args:
            args: Configuration arguments
            param_list: List of model parameters
            train_losses: List of training losses
        
        Returns:
            tuple: (sigma weights, final weighted parameters)
        """
        train_losses = torch.stack(train_losses)
        sigma = F.softmax(-train_losses / args.temp_scaling, dim=0)
        final_weights = linear_combination(param_list, sigma=sigma)
        return sigma, final_weights

    def outputs(self, args, batch):
        """Compute ensemble outputs and weights.
        
        Args:
            args: Configuration arguments
            batch: Tuple of (input, target)
        
        Returns:
            tuple: (final weights, sigma weights)
        """
        train_losses = []
        param_list = []
        train_input, train_target = batch
        
        # Process all models in eval mode
        with torch.no_grad():  # Optimization: disable gradients during forward pass
            for model in self.models:
                model.eval()
                logits = model(train_input)
                train_loss = F.cross_entropy(logits, train_target)
                train_losses.append(train_loss)
                param_list.append(OrderedDict(model.meta_named_parameters()))

        sigmas, weights = self._compute_final_weights(args, param_list, train_losses)
        return weights, sigmas