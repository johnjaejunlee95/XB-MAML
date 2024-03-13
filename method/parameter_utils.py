from collections import OrderedDict
import torch 


def average_params(param_list):
    mean_params = OrderedDict()

    for param_key in param_list[0].keys():
        mean_params[param_key] = torch.stack([params[param_key] for params in param_list]).mean(dim=0)

    return mean_params


def sample_model(args, parameters):
    
    model_dict = OrderedDict()
    for i, (key, parameters) in enumerate(parameters.items()):
        print(key)
        param = parameters + args.lam*torch.randn_like(parameters)
        param.requires_grad_(True)
        model_dict[key] = param
        
    return model_dict


def orthogonal_regularization(args, param_list):

    orthogonal_loss = []
    for i, model in enumerate(param_list):
        other_params = [params for j, params in enumerate(param_list) if i != j]
        loss = torch.tensor(0., requires_grad=True).cuda()
        
        model_params = torch.cat([p.flatten() for p in model.parameters()]).cuda()
        
        for other_model in other_params:

            other_model_params = torch.cat([p.flatten() for p in other_model.parameters()]).cuda()
            dot_product = torch.dot(model_params.view(-1), other_model_params.view(-1))
            
            loss += args.regularizer * torch.abs(dot_product)/(len(param_list) - 1)
        orthogonal_loss.append(loss)
    
    return orthogonal_loss


def projection_error(finetuned_params, subspace_param):
    
    finetuned_vector = torch.cat([p.flatten().detach() for key, p in finetuned_params.items()]).detach().cpu()

    subspace_flatten = []
    for i in range(len(subspace_param)):
        subspace_flatten.append(torch.cat([p.flatten().detach() for key, p in subspace_param[i].named_parameters()]))
    hyperplane = torch.stack(subspace_flatten).detach().cpu() # space #
        
    hyperplane = hyperplane / torch.norm(hyperplane, p=2, dim=1, keepdim=True)
    proj_inv = torch.inverse(torch.matmul(hyperplane, hyperplane.t()))
    projection = torch.matmul(torch.transpose(hyperplane, 0, 1), torch.matmul(proj_inv, torch.matmul(hyperplane, finetuned_vector.view(-1,1))))
    
    error = ((finetuned_vector.view(1,-1) - projection.view(1,-1))**2).sum().sqrt().detach().cpu()/((finetuned_vector.view(1,-1)**2).sum().sqrt().detach().cpu())
    
    return error

    
def linear_combination(model_parameters=None, sigma=None):
        
    final_weights = OrderedDict()
    for i in range(len(model_parameters)):
        for key in model_parameters[i].keys():
            if key not in final_weights.keys():
                final_weights[key] = model_parameters[i][key]*sigma[i].item()
                final_weights[key].requires_grad_(True)
            else:
                final_weights[key] = final_weights[key] + model_parameters[i][key]*sigma[i].item()
                final_weights[key].requires_grad_(True)

    return final_weights

