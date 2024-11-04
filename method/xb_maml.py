# from copy import deepcopy

# import torch
# import numpy as np
# from torch import nn

# from utils.utils import *
# from utils import add_maml
# from .gradient_utils import *
# from .parameter_utils import *


# class XB_MAML(nn.Module):
#     def __init__(self, model, ensemble_model, optimizer, meta_generator, moving_avg):

#         super(XB_MAML, self).__init__()

#         self.model = model
#         self.ensemble_model = ensemble_model
#         self.loss = nn.CrossEntropyLoss()
#         self.meta_generators = meta_generator
#         self.finetuning = Finetuning()
#         self.optimizer = optimizer
#         self.moving_avg = moving_avg
       
#     def forward(self, epoch, args, loader, logger):
        
#         loss_list = []
#         error_norm_lists = []
#         acc_list = []
#         ortho_sum = 0.
        
#         gradients = [{} for _ in range(len(self.ensemble_model.model))] # for multiple model
        
#         for i in range(args.batch_size):
            
#             batch, _ = next(loader)
#             n_way, n_support, n_query, train_target, test_target = get_basic_expt_info(args)
#             train_input, test_input = split_support_query(batch, n_support=n_support, n_query=n_query, n_way=n_way)
            
#             batch_cuda = [train_input, train_target]
            
#             init_params, sigmas = self.ensemble_model.outputs(args, batch_cuda)
#             self.model.load_state_dict(init_params)
                        
#             self.model.train()
#             self.model.zero_grad()
            
#             finetuned_params, _ = self.finetuning.maml_inner_adapt(args, self.model, train_input, train_target, args.update_step, first_order=args.first_order)
            
#             logits = self.model(test_input, finetuned_params)
#             loss_test = self.loss(logits, test_target)
          
#             acc = count_acc(logits, test_target)
            
#             # Gradient Accumulation for second order derivatives
#             grads = torch.autograd.grad(loss_test, self.model.parameters())
#             for i in range(len(self.ensemble_model.model)):
#                 for grad, (key, param) in zip(grads, self.ensemble_model.model[i].meta_named_parameters()):
#                     if key not in gradients[i].keys():
#                         gradients[i][key] = (grad.clone()*(sigmas[i].item())/args.batch_size)
#                     else:
#                         gradients[i][key] = gradients[i][key] + (grad.clone()*(sigmas[i].item())/args.batch_size)
                
#             error_norm = projection_error(finetuned_params, self.ensemble_model.model)
#             error_norm_lists.append(error_norm)
           
#             loss_list.append(loss_test.clone().cpu())
#             acc_list.append(acc)


#         # add orthogonal regularization
#         self.ensemble_model.optimizers.zero_grad() 
#         if args.num_maml != 0:
#             orthogonal_regs = orthogonal_regularization(args, self.ensemble_model.model)
#             ortho_sum += torch.sum(torch.stack(orthogonal_regs)).item()
#             for i in range(len(self.ensemble_model.model)):
#                 self.ensemble_model.model[i].zero_grad()
#                 ortho_grad = torch.autograd.grad(orthogonal_regs[i], self.ensemble_model.model[i].parameters(), retain_graph=True)
#                 for grad, ortho_reg_grad, param_n in zip(gradients[i].values(), ortho_grad, self.ensemble_model.model[i].parameters()):
#                     param_n.grad = grad + ortho_reg_grad     
#         else:
#             for grad, param_n in zip(gradients[i].values(), self.ensemble_model.model[i].parameters()):
#                 param_n.grad = grad
        
#         self.ensemble_model.optimizers.step()     
                
        
#         loss_ = torch.stack(loss_list).mean().clone().cpu() + ortho_sum
#         acc = np.array(acc_list).mean()

#         error_norm_lists = torch.stack(error_norm_lists).mean().clone().cpu()
        
#         self.moving_avg.add_value(error_norm_lists.item())
#         if (self.moving_avg.check_improvement()):
#             logger.info("Converged at epoch {}".format(epoch+1))
#             add_maml.Add_MAML(self.model).add_maml(args, self.ensemble_model, self.moving_avg, self.meta_generators)
        
#         return acc, loss_

    
#     def validation(self, args, loader, logger):
#         avg_dict = {x: Average(x) for x in self.meta_generators.stat_keys}
        
#         model = deepcopy(self.model).cuda()
        
#         for i, (batch, _) in enumerate(loader):
            
#             cluster_id = self.meta_generators.m_sampler['valid'].cluster_samples[i]
#             cluster_name = self.meta_generators.m_sampler['valid'].cluster_datasets[cluster_id]
                        
#             n_way, n_support, n_query, train_target, test_target = get_basic_expt_info(args)
#             train_input, test_input = split_support_query(batch, n_support=n_support, n_query=n_query, n_way=n_way)
            
#             batch_cuda = [train_input, train_target]
            
#             weights, _ = self.ensemble_model.outputs(args, batch_cuda)
#             model.load_state_dict(weights)
            
#             model.eval()
#             params, _ = self.finetuning.maml_inner_adapt(args, model, train_input, train_target, args.update_step_test, first_order=True)
            
#             logits = model(test_input, params)
    
#             with torch.no_grad():
#                 loss_ = self.loss(logits, test_target)
#                 acc = count_acc(logits, test_target)

#             avg_dict["loss"].add(loss_.item())
#             avg_dict["{}_loss".format(self.meta_generators.n_cluster)].add(loss_.item())
#             avg_dict["{}_loss".format(cluster_name)].add(loss_.item())
            
#             avg_dict["acc"].add(acc)
#             avg_dict["{}_acc".format(self.meta_generators.n_cluster)].add(acc)
#             avg_dict["{}_acc".format(cluster_name)].add(acc)
                    
        
#         avg_acc, avg_loss = log_acc(args, avg_dict, self.meta_generators, 'validation', logger)
#         logger.info("====================================================")
        
#         return avg_acc, avg_loss
    
    
# def main():
#     pass

# if __name__ == "__main__":
#     main()

from copy import deepcopy

import torch
import numpy as np
from torch import nn

from utils.utils import *
from utils import add_maml
from .gradient_utils import *
from .parameter_utils import *


class XB_MAML(nn.Module):
    def __init__(self, model, ensemble_model, optimizer, meta_generator, moving_avg):

        super(XB_MAML, self).__init__()

        self.model = model
        self.ensemble_model = ensemble_model
        self.loss = nn.CrossEntropyLoss()
        self.meta_generators = meta_generator
        self.finetuning = Finetuning()
        self.optimizer = optimizer
        self.moving_avg = moving_avg
       
    def forward(self, epoch: int, args, loader, logger):
        
        loss_list = []
        error_norm_lists = []
        acc_list = []
        ortho_sum = 0.
        
        gradients = [{} for _ in range(len(self.ensemble_model.model))]
        
        for batch_idx in range(args.batch_size):
            
            batch, _ = next(loader)
            n_way, n_support, n_query, train_target, test_target = get_basic_expt_info(args)
            train_input, test_input = split_support_query(batch, n_support=n_support, n_query=n_query, n_way=n_way)
            
            init_params, sigmas = self.ensemble_model.outputs(args, [train_input, train_target])
            self.model.load_state_dict(init_params)
                        
            with torch.set_grad_enabled(not args.first_order):
                finetuned_params, _ = self.finetuning.maml_inner_adapt(
                    args, self.model, train_input, train_target, 
                    args.update_step, first_order=args.first_order
                )
            
            logits = self.model(test_input, finetuned_params)
            loss_test = self.loss(logits, test_target)
          
            acc = count_acc(logits, test_target)
            
            grads = torch.autograd.grad(loss_test, self.model.parameters())
            for model_idx in range(len(self.ensemble_model.model)):
                sigma_factor = sigmas[model_idx].item() / args.batch_size
                for grad, (key, _) in zip(grads, self.ensemble_model.model[model_idx].meta_named_parameters()):
                    gradients[model_idx][key] = gradients[model_idx].get(key, 0) + (grad.clone() * sigma_factor)
            
            error_norm = projection_error(finetuned_params, self.ensemble_model.model)
            error_norm_lists.append(error_norm)
           
            loss_list.append(loss_test.detach().cpu())
            acc_list.append(acc)

        self.ensemble_model.optimizers.zero_grad()
        if args.num_maml != 0:
            orthogonal_regs = orthogonal_regularization(args, self.ensemble_model.model)
            ortho_sum = torch.sum(torch.stack(orthogonal_regs)).item()
            
            for model_idx in range(len(self.ensemble_model.model)):
                self.ensemble_model.model[model_idx].zero_grad()
                ortho_grad = torch.autograd.grad(
                    orthogonal_regs[model_idx], 
                    self.ensemble_model.model[model_idx].parameters(), 
                    retain_graph=True
                )
                
                for grad, ortho_reg_grad, param in zip(
                    gradients[model_idx].values(), 
                    ortho_grad, 
                    self.ensemble_model.model[model_idx].parameters()
                ):
                    param.grad = grad + ortho_reg_grad
        else:
            for model_idx in range(len(self.ensemble_model.model)):
                for grad, param in zip(
                    gradients[model_idx].values(), 
                    self.ensemble_model.model[model_idx].parameters()
                ):
                    param.grad = grad
        
        self.ensemble_model.optimizers.step()     
                
        
        loss_ = torch.stack(loss_list).mean().cpu() + ortho_sum
        acc = np.mean(acc_list)

        error_norm_mean = torch.stack(error_norm_lists).mean().cpu()
        
        self.moving_avg.add_value(error_norm_mean.item())
        if self.moving_avg.check_improvement():
            logger.info(f"Converged at epoch {epoch+1}")
            add_maml.Add_MAML(self.model).add_maml(
                args, self.ensemble_model, self.moving_avg, self.meta_generators
            )
        
        return acc, loss_

    # @torch.no_grad()
    def validation(self, args, loader, logger):
        avg_dict = {x: Average(x) for x in self.meta_generators.stat_keys}
        model = deepcopy(self.model).cuda()
        
        for i, (batch, _) in enumerate(loader):
            cluster_id = self.meta_generators.m_sampler['valid'].cluster_samples[i]
            cluster_name = self.meta_generators.m_sampler['valid'].cluster_datasets[cluster_id]
            
            n_way, n_support, n_query, train_target, test_target = get_basic_expt_info(args)
            train_input, test_input = split_support_query(
                batch, n_support=n_support, n_query=n_query, n_way=n_way
            )
            
            # Get model weights
            weights, _ = self.ensemble_model.outputs(args, [train_input, train_target])
            model.load_state_dict(weights)
            model.eval()
            
            # Fine-tune
            params, _ = self.finetuning.maml_inner_adapt(
                args, model, train_input, train_target, 
                args.update_step_test, first_order=True
            )
            
            # Evaluate
            logits = model(test_input, params)
            loss_ = self.loss(logits, test_target)
            acc = count_acc(logits, test_target)

            # Update metrics
            cluster_prefix = f"{self.meta_generators.n_cluster}"
            for metric, value in [("loss", loss_.item()), ("acc", acc)]:
                avg_dict[metric].add(value)
                avg_dict[f"{cluster_prefix}_{metric}"].add(value)
                avg_dict[f"{cluster_name}_{metric}"].add(value)
        
        avg_acc, avg_loss = log_acc(args, avg_dict, self.meta_generators, 'validation', logger)
        logger.info("====================================================")
        
        return avg_acc, avg_loss
    
    
def main():
    pass

if __name__ == "__main__":
    main()
