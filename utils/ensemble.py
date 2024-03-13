import torch
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules import MetaModule
from method.parameter_utils import linear_combination

class Ensemble_Model(MetaModule):
    def __init__(self):
        super(Ensemble_Model, self).__init__()

        self.optimizers = None
        self.model = []

    def add_additional_model(self, args, new_network, status='train'):
       
        self.model.append(new_network)
            
        if status == 'train':
            if self.optimizers is None:
                self.optimizers = torch.optim.Adam(new_network.parameters(), lr=args.meta_lr)
            else:
                self.optimizers.add_param_group({'params': new_network.parameters(), 'lr': args.meta_lr})
        else:
            pass
            
    def _compute_final_weights(self, args, param_list, train_losses=None):
        sigma = F.softmax(-torch.stack(train_losses) / args.temp_scaling, dim=0)
        final_weights = linear_combination(param_list, sigma=sigma)
        return sigma, final_weights

    def outputs(self, args, batch=None):
                
        train_losses = []
        param_list = []
        train_input, train_target = batch
        
        for i in range(len(self.model)):
            self.model[i].eval()
            logits = self.model[i](train_input)
            train_loss = F.cross_entropy(logits, train_target)
            self.model[i].zero_grad()
            train_losses.append(train_loss)
            param_list.append(OrderedDict(self.model[i].meta_named_parameters()))

        sigmas, weights = self._compute_final_weights(args, param_list, train_losses)

        return weights, sigmas
