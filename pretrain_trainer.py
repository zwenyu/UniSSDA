import torch
import torch.nn.functional as F
import shutil
import os
import wandb
import collections
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter, fix_randomness, copy_Files, starting_logs
from dataloader.dataloader import return_classlist, pretrain_data_generator, get_select_class

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class pretrain(object):
    """
   This class contain the pretraining functions
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.da_setting = args.da_setting
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.sampling = args.sampling
        self.num_shots = args.num_shots  # for k-shot sampling

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description
        self.wandb_dir = args.wandb_dir
        self.wandb_project = args.wandb_project
        self.wandb_entity = args.wandb_entity
        self.wandb_tag = args.wandb_tag

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.backbone_save_dir = args.backbone_save_dir
        self.dataset_txt_path = os.path.join(args.data_path, self.dataset)  # txt files
        self.im_dir = os.path.join(args.data_root, self.dataset)  # image files

        # Specify runs
        self.num_seeds = args.num_seeds

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.train_params,
                                **self.hparams_class.backbone_hparams[self.backbone],
                                **self.hparams_class.alg_hparams[self.da_method],
                                }

    def train(self):
        run_name = f"{self.run_description}"
        # set default hyperparameters
        os.makedirs(self.wandb_dir, exist_ok=True)
        wandb.init(config=self.default_hparams, mode="online", name=run_name, tags=self.wandb_tag,
            project=self.wandb_project, entity=self.wandb_entity, dir=self.wandb_dir)

        # wandb.config logs all hyperparameters
        self.hparams = wandb.config
        #! for debugging
        if 'debug' in self.wandb_tag:
            wandb.config.update({"num_steps": 500}, allow_val_change=True)
        # Logging
        self.backbone_log_dir = os.path.join(self.backbone_save_dir, self.experiment_description, run_name)
        os.makedirs(self.backbone_log_dir, exist_ok=True)
        copy_Files(self.backbone_log_dir)  # save a copy of training files
        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        src_class = get_select_class(self.dataset, True, self.da_setting)
        trg_class = get_select_class(self.dataset, False, self.da_setting)
        self.metrics = {}

        for i in scenarios:
            src_domain = i[0]
            trg_domain = i[1]

            for run_id in range(self.num_seeds):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.backbone_log_dir,
                                                                   src_domain, trg_domain, run_id)

                # Load data
                self.pretrain_load_data(src_domain, trg_domain)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                backbone_fe = get_backbone_class(self.backbone)

                algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams,
                                            self.device, src_class, trg_class)
                algorithm.to(self.device)

                # Track losses at every step for each scenario_run
                scenario_run_step_losses = {}
                # Track accuracy at every epoch for each scenario_run
                scenario_run_step_acc = {}

                scenario_run = f"pretrain_{src_domain}_to_{trg_domain}_run{run_id}"

                # skip training if checkpoint already exists
                last_checkpoint_path = os.path.join(self.home_path, self.scenario_log_dir, 'last_checkpoint.pt')
                if os.path.exists(last_checkpoint_path):
                    print(f'{last_checkpoint_path} exists, skipping training. Results files will be regenerated.')
                else:
                    algorithm.train()
                    src_train_iter = iter(self.src_train_dl)
                    trg_train_iter = iter(self.trg_train_dl)

                    len_src_train = len(self.src_train_dl)
                    print('len(self.src_train_dl)')
                    print(len(self.src_train_dl))
                    len_trg_train = len(self.trg_train_dl)
                    print('len(self.trg_train_dl)')
                    print(len(self.trg_train_dl))

                    for step in range(self.hparams["num_steps"]):
                        # Average meters
                        if (step % self.hparams['eval_interval'] == 0):
                            loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                        # Restart data loaders
                        if step % len_src_train == 0 and step > 0:
                            src_train_iter = iter(self.src_train_dl)
                        if step % len_trg_train == 0 and step > 0:
                            trg_train_iter = iter(self.trg_train_dl)

                        src_train_data = next(src_train_iter)
                        trg_train_data = next(trg_train_iter)

                        im_data_src = src_train_data[0].reshape(
                            (-1,) + src_train_data[0].shape[2:]).to(self.device)
                        gt_labels_src = src_train_data[2].reshape(
                            (-1,) + src_train_data[2].shape[2:]).to(self.device)
                        im_data_trg = trg_train_data[0].reshape(
                            (-1,) + trg_train_data[0].shape[2:]).to(self.device)
                        gt_labels_trg = trg_train_data[2].reshape(
                            (-1,) + trg_train_data[2].shape[2:]).to(self.device)
                        # print("im_data_src shape: ", im_data_src.shape)
                        # print("im_data_trg shape: ", im_data_trg.shape)
                        # Labeled source data
                        x_src = src_train_data[0]

                        losses = algorithm.update(im_data_src, im_data_trg, gt_labels_src, gt_labels_trg, step)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, x_src.size(0))
                            scenario_run_step_losses[scenario_run + '_' + key] = val
                        wandb.log(scenario_run_step_losses)

                        if step > 0 and step % self.hparams['eval_interval'] == 0:
                            # track accuracy
                            acc_src_test = self.get_accuracy(algorithm, self.src_train_dl, 'src')
                            acc_trg_test = self.get_accuracy(algorithm, self.trg_train_dl, 'trg')
                            acc_dict = {'rot_pred_acc_src': acc_src_test, 'rot_pred_acc_trg': acc_trg_test}
                            for key, val in acc_dict.items():
                                scenario_run_step_acc[scenario_run + '_' + key] = val
                            wandb.log(scenario_run_step_acc)

                            self.logger.debug(f'[Step : {step}/{self.hparams["num_steps"]}]')
                            for key, val in loss_avg_meters.items():
                                self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                            for key, val in acc_dict.items():
                                self.logger.debug(f'{key}\t: {val:2.4f}')
                            self.logger.debug('-------------------------------------')

                    save_path = os.path.join(self.home_path, self.scenario_log_dir, 'last_checkpoint.pt')
                    torch.save({
                        'Train Step': step,
                        'G_state_dict': algorithm.feature_extractor.state_dict(),
                        'F2_state_dict': algorithm.classifier.state_dict(),
                        'optimizer_g_state_dict': algorithm.optimizer_g.state_dict(),
                        'optimizer_f_state_dict': algorithm.optimizer_f.state_dict(),
                    }, save_path)

    def pretrain_load_data(self, src_domain, trg_domain):
        self.class_list = return_classlist(os.path.join(self.dataset_txt_path, 'train_' + src_domain + '.txt'))
        self.src_train_dl, self.src_val_dl = pretrain_data_generator(self.dataset_txt_path, src_domain, self.hparams,
                                                                     self.im_dir, True, self.da_setting,self.backbone)
        self.trg_train_dl, self.trg_val_dl = pretrain_data_generator(self.dataset_txt_path, trg_domain, self.hparams,
                                                                     self.im_dir, False, self.da_setting,self.backbone)

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(self.wandb_tag), hparams_class()


    def get_accuracy(self, algorithm, loader, domain_type):
        feature_extractor = algorithm.feature_extractor.to(self.device)
        classifier = algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        trg_pred_labels = np.array([])
        trg_true_labels = np.array([])

        with torch.no_grad():
            # img, target, self.im_paths[index], index
            for test_data in loader:
                data = test_data[0].reshape(
                    (-1,) + test_data[0].shape[2:]).cuda(non_blocking=True)
                label = test_data[2].reshape(
                    (-1,) + test_data[2].shape[2:]).cuda(non_blocking=True)
                # data = data.to(self.device)
                # label = label.to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features, domain_type=domain_type)
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                trg_pred_labels = np.append(trg_pred_labels, pred.cpu().numpy())
                trg_true_labels = np.append(trg_true_labels, label.data.cpu().numpy())

        feature_extractor.train()
        classifier.train()

        acc = accuracy_score(trg_true_labels, trg_pred_labels)
        return acc * 100