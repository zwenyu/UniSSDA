import os
import argparse
import warnings
from trainer import cross_domain_trainer
from pretrain_trainer import pretrain
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs', type=str, help='Directory containing all experiments logs')
parser.add_argument('--experiment_description', default='Proposed', type=str, help='Name of your experiment, used only for local logging')
parser.add_argument('--run_description',        default='test', type=str, help='Name of your runs, used for local logging and on wandb')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='Proposed', type=str, help='Name of algorithm to use')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./data/txt', type=str, help='Folder containing dataset image paths')
parser.add_argument('--data_root',              default=r'./data', type=str, help='Folder containing dataset images')
parser.add_argument('--dataset',                default='office_home', type=str, choices=['office_home', 'domain_net'], help='Dataset of choice')
parser.add_argument('--da_setting',             default='closed', type=str, help='DA setting of choice')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='Resnet34', type=str, help='Backbone network of choice e.g. AlexNetBase, Resnet34')
parser.add_argument('--backbone_save_dir',      default='experiments_rot_pred', type=str, help='Directory containing all experiments logs for rotation pretraining')

# ========= Experiment settings ===============
parser.add_argument('--num_seeds',              default=3, type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0', type=str, help='cpu or cuda')
parser.add_argument('--sampling',               default='kshot', type=str, help='Sampling strategy used to select labeled target data')
parser.add_argument('--num_shots',              default='3', type=int, help='Number of labels per class')
parser.add_argument('--warm_steps',              default='500', type=int, help='Number of warm up steps')
parser.add_argument('--metric_to_maximize',     default="acc_trg_val", type=str, choices=['acc_src_val', 'acc_trg_val'], help='Metric for model selection')

# ======== WandB settings =====================
parser.add_argument('--wandb_dir',              default='wandb', type=str, help='Directory to store Wandb metadata')
parser.add_argument('--wandb_project',          default='TEST_SOMETHING', type=str, help='Project name in Wandb')
parser.add_argument('--wandb_entity',           type=str, help='Entity name in Wandb (can be left blank if there is a default entity)')
parser.add_argument('--wandb_tag',              default=[], type=str, nargs='*', help='tags for Wandb runs e.g. subset to run subset of domains, debug to run subset of domains for 500 steps')

args = parser.parse_args()
args.wandb_tag += [args.dataset, args.backbone, args.da_method, args.da_setting, args.sampling, f'{args.num_shots}shots']
args.wandb_tag += [args.data_path.split('/')[-1]]

if __name__ == "__main__":

    if args.da_method == 'pretrain':
        trainer = pretrain(args)
    else:
        trainer = cross_domain_trainer(args)

    trainer.train()