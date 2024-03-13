import random
import datetime as dt
import torch
import numpy as np
from copy import deepcopy

from model import model_selection
from utils.validate import Validation
from utils.utils import * 
from utils.args import parse_args
from utils.ensemble import Ensemble_Model
from utils.meta_generator import MetaDatasetsGenerator


def main(args):
    # Seed initialization
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Argument initialization
    args.best_loss = 100
    args.best_acc = 0
    args.num_maml = 0
    args.check = False
    args.current_epoch = 0

    # MetaDatasetsGenerator initialization
    meta_generators = MetaDatasetsGenerator(args)
    dataloader = meta_generators.m_dataloader

    # Ensemble model initialization
    ensemble_model = Ensemble_Model()
    maml, model = model_selection(args, meta_generators, ensemble_model)
    ensemble_model.add_additional_model(args, model)
    
    args.avg_dict = {x: Average(x) for x in meta_generators.stat_keys}

    # Validation initialization
    validation = Validation(maml, dataloader['valid'], model)

    # Cycle through the train loader
    train_loader = cycle(dataloader['train'])
    myLogger.info("# of Parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    
    # Main training loop
    for epoch in range(args.epochs):
        args.current_epoch += 1
        cluster_id = meta_generators.m_sampler['train'].cluster_samples[epoch * args.batch_size]
        cluster_name = meta_generators.m_sampler['train'].cluster_datasets[cluster_id]

        # Forward pass through MAML
        acc, loss = maml.forward(epoch, args, train_loader, myLogger)

        # Update average metrics
        args.avg_dict["{}_loss".format(cluster_name)].add(loss.item())
        args.avg_dict["{}_acc".format(cluster_name)].add(acc)
        args.avg_dict["{}_loss".format(meta_generators.n_cluster)].add(loss.item())
        args.avg_dict["{}_acc".format(meta_generators.n_cluster)].add(acc)
        args.avg_dict["loss"].add(loss.item())
        args.avg_dict["acc"].add(acc)

        # Logging and saving checkpoints
        if (epoch + 1) % 1000 == 0:
            log_acc(args, args.avg_dict, meta_generators, 'training', myLogger)
            torch.save({'model': [deepcopy(models).to('cpu') for models in ensemble_model.model],
                        'optimizer': ensemble_model.optimizers.state_dict()},
                       SAVE_PATH + CKPT_PATH)

        # Validation
        if (epoch + 1) % 10000 == 0 or epoch == 0:
            validation.validation_acc(args, myLogger, ensemble_model)

        # Learning rate scheduling
        if (epoch + 1) % (args.epochs // 4) == 0:
            args.meta_lr *= 0.8
            for i, param_group in enumerate(ensemble_model.optimizers.param_groups):
                param_group['lr'] = args.meta_lr

    print("Training Done")

if __name__ == "__main__":
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    
    args = parse_args()
    RANDOM_SEED = random.randint(0, 1000)

    SAVE_PATH = "./save/ckpt/" # your own path
    CKPT_PATH = "XB_MAML_5-{}_{}_{}_version_{}.pt".format(str(args.num_shots), args.datasets, args.model, args.version)

    x = dt.datetime.now()
    base_name = 'XB_MAML'

    dataset_info = '{}_{}way_{}shot_{}_version_{}'.format(args.datasets, str(args.num_ways), str(args.num_shots), str(args.model), args.version)

    name = f'{dataset_info}'

    myLogger = log_(name)
    myLogger.info(name)
    myLogger.info("Start Training!!")
    main(args)
    