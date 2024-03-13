import torch
from copy import deepcopy

class Validation():
    def __init__(self, maml, dataloader, model):
        super(Validation, self).__init__()
        
        self.maml = maml
        self.dataloader = dataloader
        self.model = model

    def validation_acc(self, args, logger, ensemble_model):
        
        SAVE_PATH = "./save/ckpt/" # your own path
        
        fmt = "{}-{}_{}_{}_version_{}_best.pt"

        CKPT_PATH = fmt.format("XB_MAML_5", args.num_shots, args.datasets, args.model, args.version)

        acc, loss = self.maml.validation(args, self.dataloader, logger)
                    
        if args.best_acc <= acc:
            args.best_acc = acc
            args.best_loss = loss
            
            torch.save({'model':[deepcopy(models).to('cpu') for models in ensemble_model.model],
                       'optimizer': ensemble_model.optimizers.state_dict(),},
                        SAVE_PATH + CKPT_PATH)

        # logger.info("Version:{} Current Acc: {:.2f}, Best Acc: {:.2f}".format(args.version, acc, args.best_acc))
        # logger.info("====================================================")
        
            
 