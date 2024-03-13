import torch
from torch import nn

from model.conv4 import Conv_Model
from model.resnet import ResNet12
from utils.ensemble import Ensemble_Model
from utils.utils import get_basic_expt_info, split_support_query, count_acc, Average
from .gradient_utils import Finetuning

class XB_MAML_Test(nn.Module):
    def __init__(self, meta_generator):
        super(XB_MAML_Test, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.finetuning = Finetuning()
        self.ensemble_model = Ensemble_Model()
        self.meta_generator = meta_generator

    def forward(self, args, loader, model_parameters):
        acc = 0.
        avg_dict = {x: Average(x) for x in self.meta_generator.stat_keys}
        features = {}

        for i in range(len(model_parameters)):
            model_parameters[i] = model_parameters[i].cuda()
            self.ensemble_model.add_additional_model(args, model_parameters[i], 'test')

        feature_size = args.filter_size*5*5
        if args.model == 'conv4':
            model = Conv_Model(args.imgc, args.num_ways, args.filter_size, feature_size, drop_p=args.dropout).cuda().eval()
        else:
            model = ResNet12(args.num_ways, drop_p=args.dropout).cuda().eval()
        
        power_set = {i: {} for i in range(len(model_parameters))}
        for n, (batch, _) in enumerate(loader['test']):
            cluster_id = self.meta_generator.m_sampler['test'].cluster_samples[n]
            cluster_name = self.meta_generator.m_sampler['test'].cluster_datasets[cluster_id]

            n_way, n_support, n_query, y_support, y_query = get_basic_expt_info(args)
            x_support, x_query = split_support_query(batch, n_support=n_support, n_query=n_query, n_way=n_way)

            train_input = x_support.cuda()
            train_target = y_support.cuda()
            test_input = x_query.cuda()
            test_target = y_query.cuda()
            batch_cuda = [train_input, train_target]

            
            weights, alphas = self.ensemble_model.outputs(args, batch_cuda)
            model.load_state_dict(weights)
            model.eval()
            params, _ = self.finetuning.maml_inner_adapt(args, model, train_input, train_target, args.update_step_test, first_order=True)
            model.eval()
            logits = model(test_input, params)

            new_params = torch.cat([p.flatten() for p in params.values()]).detach().cpu()
            if cluster_name not in features:
                features[cluster_name] = []
            features[cluster_name].append(new_params)
            
            for i in range(len(model_parameters)):
                if cluster_name not in power_set[i]:
                    power_set[i][cluster_name] = alphas[i].detach().cpu()/args.max_test_task
                power_set[i][cluster_name] += alphas[i].detach().cpu()/args.max_test_task

            loss_ = self.loss(logits, test_target)
            acc = count_acc(logits, test_target)

            avg_dict["{}_loss".format(cluster_name)].add(loss_.item())
            avg_dict["{}_acc".format(cluster_name)].add(acc)
            avg_dict["loss"].add(loss_.item())
            avg_dict["{}_loss".format(self.meta_generator.n_cluster)].add(loss_.item())
            avg_dict["acc"].add(acc)
            avg_dict["{}_acc".format(self.meta_generator.n_cluster)].add(acc)

        cluster_acc = {}
        cluster_loss = {}
        for cluster_name in self.meta_generator.cluster_name:
            acc = 100 * avg_dict["{}_acc".format(cluster_name)].item()
            loss = avg_dict["{}_loss".format(cluster_name)].item()
            cluster_acc[cluster_name] = acc
            cluster_loss[cluster_name] = loss
            print("cluster {}: acc = {:.4f}, loss = {:.4f}".format(cluster_name, acc, loss))
        avg_acc = 100 * avg_dict["{}_acc".format(self.meta_generator.n_cluster)].item()
        avg_loss = avg_dict["{}_loss".format(self.meta_generator.n_cluster)].item()
        print("Average acc = {:.4f}".format(avg_acc))
        print("Average Loss = {:.4f}".format(avg_loss))

        return avg_acc, avg_loss, cluster_acc, cluster_loss, power_set, features

