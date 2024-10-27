
## The current hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class office_home():
    def __init__(self):
        super(office_home, self).__init__()
        # Common training parameters
        self.train_params = {
            'num_steps': 5000,
            'eval_interval': 500,
            # optimizer and scheduler
            'rampup_length': 20000,
            'rampup_coef': 30.0,
            'weight_decay': 5e-4,
            'gamma': 1e-4,
        }

        # Backbone-specific hyperparamters
        self.backbone_hparams = {
            'AlexNetBase': {'batch_size': 32},
            'SwinTiny': {'batch_size': 16},
            'Resnet18': {'batch_size': 32},
            'Resnet34': {'batch_size': 24},
        }

        # Algorithm-specific hyperparameters
        self.alg_hparams = {
            'Baseline': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05},
            'CDAC': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'topk': 5, 'threshold': 0.95, 'temp': 0.05},
            'pretrain': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05,
                        'cls_layers': '', 'cls_normalize': True, 'cls_bias': False},
            'PAC': {'lr': 1, 'lr_f': 0.01, 'multi': 0.001, 'temp': 0.05,
                        'cls_layers': '', 'cons_wt': 1., 'cons_threshold': 0.9},
            'AdaMatch': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05, 'tau': 0.9},
            'DST': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05, 'tau': 0.9},
            'Proposed': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05, 'tau': 0.9},
        }


class domain_net():
    def __init__(self):
        super(domain_net, self).__init__()
        # Common training parameters
        self.train_params = {
            'num_steps': 5000,
            'eval_interval': 500,
            # optimizer and scheduler
            'rampup_length': 20000,
            'rampup_coef': 30.0,
            'weight_decay': 5e-4,
            'gamma': 1e-4,
        }

        # Backbone-specific hyperparamters
        self.backbone_hparams = {
            'AlexNetBase': {'batch_size': 32},
            'SwinTiny': {'batch_size': 16},
            'Resnet18': {'batch_size': 32},
            'Resnet34': {'batch_size': 24},
        }

        # Algorithm-specific hyperparameters
        self.alg_hparams = {
            'Baseline': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05},
            'CDAC': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'topk': 5, 'threshold': 0.95, 'temp': 0.05},
            'pretrain': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05,
                        'cls_layers': '', 'cls_normalize': True, 'cls_bias': False},
            'PAC': {'lr': 1, 'lr_f': 0.01, 'multi': 0.001, 'temp': 0.05,
                        'cls_layers': '', 'cons_wt': 1., 'cons_threshold': 0.9},
            'AdaMatch': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05, 'tau': 0.9},
            'DST': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05, 'tau': 0.9},
            'Proposed': {'lr': 0.01, 'lr_f': 1.0, 'multi': 0.1, 'temp': 0.05, 'tau': 0.9},
        }