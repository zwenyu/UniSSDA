import os
import re
import wandb
import warnings
import collections
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import collections
from sklearn.metrics import accuracy_score

from dataloader.dataloader import data_generator, get_select_class, return_classlist
from dataloader.target_selection_dataloader import sampling_data_generator
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class cross_domain_trainer(object):
    """
   This class contain the main training functions
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.da_setting = args.da_setting
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.sampling = args.sampling
        self.num_shots = args.num_shots  # for k-shot sampling
        self.warm_steps = args.warm_steps
        self.metric_to_maximize = args.metric_to_maximize

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
        self.create_save_dir()

        # Specify runs
        self.num_seeds = args.num_seeds

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.train_params,
                                **self.hparams_class.backbone_hparams[self.backbone],
                                **self.hparams_class.alg_hparams[self.da_method]
                                }
        self.default_hparams['warm_steps'] = self.warm_steps

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
            wandb.config.update({"num_steps": 2000}, allow_val_change=True)
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        self.src_class = get_select_class(self.dataset, True, self.da_setting)
        self.trg_class = get_select_class(self.dataset, False, self.da_setting)

        self.metrics = {ckpt_name: {} for ckpt_name in ['last_checkpoint', 'best_acc_src_val_checkpoint', 'best_acc_trg_val_checkpoint']}
        for i in scenarios:
            src_domain = i[0]
            trg_domain = i[1]

            for run_id in range(self.num_seeds):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_domain, trg_domain, run_id)

                # Load data
                self.load_data(src_domain, trg_domain, seed=run_id)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                backbone_fe = get_backbone_class(self.backbone)
                # load pretrained backbone (for PAC method)
                backbone_path = os.path.join(self.home_path, self.scenario_log_dir, 'last_checkpoint.pt')
                backbone_path = backbone_path.replace(self.save_dir, self.backbone_save_dir).replace(f'run_{run_id}', 'run_0')
                backbone_path = re.sub(f'{self.da_method}.*-{self.sampling}-{self.num_shots}', 'pretrain', backbone_path)
                self.algorithm = algorithm_class(len(self.class_list), backbone_fe, self.dataset_configs, self.hparams,
                                                self.device, self.src_class, self.trg_class, backbone_path)
                self.algorithm = self.algorithm.to(self.device)

                # Track losses at every step for each scenario_run
                scenario_run_step_losses = {}
                # Track accuracy at every epoch for each scenario_run
                scenario_run_step_acc = {}
                # Track best source and target validation accuracy
                best_acc_src_val = 0
                best_acc_trg_val = 0

                scenario_run = f"{src_domain}_to_{trg_domain}_run{run_id}"

                # training...
                src_train_iter = iter(self.src_train_dl)
                trg_train_iter = iter(self.trg_train_dl)
                trg_ul_train_iter = iter(self.trg_ul_train_dl)
                len_src_train = len(self.src_train_dl)
                len_trg_train = len(self.trg_train_dl)
                print(f'len_trg_train,{len_trg_train}')
                len_trg_ul_train = len(self.trg_ul_train_dl)
                print(f'len_trg_ul_train,{len_trg_ul_train}')

                # skip training if checkpoint already exists
                last_checkpoint_path = os.path.join(self.home_path, self.scenario_log_dir, 'last_checkpoint.pt')
                if os.path.exists(last_checkpoint_path):
                    print(f'{last_checkpoint_path} exists, skipping training. Results files will be regenerated.')
                else:
                    self.algorithm.train()
                    for step in range(1, self.hparams["num_steps"] + 1):
                        # Average meters
                        if (step == 1) or (step % self.hparams['eval_interval'] == 0):
                            loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                        # Restart data loaders
                        if step % len_src_train == 0:
                            src_train_iter = iter(self.src_train_dl)
                        if step % len_trg_train == 0:
                            trg_train_iter = iter(self.trg_train_dl)
                        if step % len_trg_ul_train == 0:
                            trg_ul_train_iter = iter(self.trg_ul_train_dl)

                        src_train_data = next(src_train_iter)
                        trg_train_data = next(trg_train_iter)
                        trg_ul_train_data = next(trg_ul_train_iter)

                        # Labeled source data
                        x_src, x_bar_src, y_src = src_train_data[0], src_train_data[3], src_train_data[1]
                        im_data_src = x_src.to(self.device)
                        im_data_bar_src = x_bar_src.to(self.device)
                        gt_labels_src = y_src.to(self.device)

                        # Labeled target data
                        x_trg, x_bar_trg, y_trg = trg_train_data[0], trg_train_data[3], trg_train_data[1]
                        im_data_trg = x_trg.to(self.device)
                        im_data_bar_trg = x_bar_trg.to(self.device)
                        gt_labels_trg = y_trg.to(self.device)

                        # Unlabeled target data
                        x_trg_ul, x_bar_trg_ul, x_bar2_trg_ul = trg_ul_train_data[0], trg_ul_train_data[3], trg_ul_train_data[4]
                        im_data_trg_ul = x_trg_ul.to(self.device)
                        im_data_bar_trg_ul = x_bar_trg_ul.to(self.device)
                        im_data_bar2_trg_ul = x_bar2_trg_ul.to(self.device)

                        losses = self.algorithm.update(im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
                                                  im_data_bar2_trg_ul, gt_labels_src, gt_labels_trg,
                                                  step, self.hparams["num_steps"])

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, x_src.size(0))
                            scenario_run_step_losses[scenario_run + '_' + key] = val
                        wandb.log(scenario_run_step_losses)

                        if (step == 1) or (step % self.hparams['eval_interval'] == 0):
                            # track accuracy
                            acc_src_val, _, _ = self.get_accuracy(self.algorithm, self.src_val_dl, 'src')
                            acc_trg_train, _, _ = self.get_accuracy(self.algorithm, self.trg_ul_train_eval_dl, 'trg')
                            acc_trg_val, _, _ = self.get_accuracy(self.algorithm, self.trg_val_dl, 'trg')
                            acc_trg_test, _, _ = self.get_accuracy(self.algorithm, self.trg_test_dl, 'trg')
                            acc_dict = {'acc_src_val': acc_src_val,
                                'acc_trg_train': acc_trg_train, 'acc_trg_val': acc_trg_val, 'acc_trg_test': acc_trg_test}
                            for key, val in acc_dict.items():
                                scenario_run_step_acc[scenario_run + '_' + key] = val
                            wandb.log(scenario_run_step_acc)

                            # track best source and target validation accuracy
                            if acc_src_val > best_acc_src_val:
                                best_acc_src_val = acc_src_val
                                save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                                                self.scenario_log_dir, self.hparams, 'best_acc_src_val_checkpoint')
                            if acc_trg_val > best_acc_trg_val:
                                best_acc_trg_val = acc_trg_val
                                save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                                                self.scenario_log_dir, self.hparams, 'best_acc_trg_val_checkpoint')

                            # logging
                            self.logger.debug(f'[Step : {step}/{self.hparams["num_steps"]}]')
                            for key, val in loss_avg_meters.items():
                                self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                            for key, val in acc_dict.items():
                                self.logger.debug(f'{key}\t: {val:2.4f}')
                            self.logger.debug('-------------------------------------')
                    save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                                    self.scenario_log_dir, self.hparams, 'last_checkpoint')

                for ckpt_name in ['last_checkpoint', 'best_acc_src_val_checkpoint', 'best_acc_trg_val_checkpoint']:
                    self.evaluate(ckpt_name)
                    self.calc_results_per_run(ckpt_name, 'train')
                    self.calc_results_per_run(ckpt_name, 'test')

        # logging metrics
        for ckpt_name in ['last_checkpoint', 'best_acc_src_val_checkpoint', 'best_acc_trg_val_checkpoint']:
            for split in ['train', 'test']:
                self.calc_overall_results(ckpt_name, split)
                wandb.log({f'{ckpt_name}_{split}_hparams': wandb.Table(
                    dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
                    allow_mixed_types=True)})
                wandb.log({f'{ckpt_name}_{split}_avg_results': wandb.Table(dataframe=self.averages_results_df[ckpt_name][split], allow_mixed_types=True)})
                wandb.log({f'{ckpt_name}_{split}_std_results': wandb.Table(dataframe=self.std_results_df[ckpt_name][split], allow_mixed_types=True)})

        average_metrics = {metric: np.mean(value) for (metric, value) in self.metrics[f'best_{self.metric_to_maximize}_checkpoint']['test'].items()}
        wandb.log(average_metrics)

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
                data, label = test_data[0], test_data[1]
                data = data.to(self.device)
                label = label.to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features, domain_type=domain_type)
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                trg_pred_labels = np.append(trg_pred_labels, pred.cpu().numpy())
                trg_true_labels = np.append(trg_true_labels, label.data.cpu().numpy())

        feature_extractor.train()
        classifier.train()

        acc = accuracy_score(trg_true_labels, trg_pred_labels)

        # percentage of samples predicted to be in shared / trg_only classes
        trg_only_class = list(set(self.trg_class) - set(self.src_class))
        pct_trg_only_true = 100 * len([label for label in trg_true_labels if label in trg_only_class]) / len(trg_true_labels)
        pct_trg_only_pred = 100 * len([label for label in trg_pred_labels if label in trg_only_class]) / len(trg_pred_labels)

        return acc * 100, pct_trg_only_true, pct_trg_only_pred

    def get_acc_group(self, algorithm, loader, classifier_choice):
        '''Returns group accuracies for diagnosis
        '''
        feature_extractor = algorithm.feature_extractor.to(self.device)
        if classifier_choice == 1:
            classifier = algorithm.classifier.to(self.device)
        elif classifier_choice == 2:
            classifier = algorithm.classifier2.to(self.device)
        domain_type = 'trg'

        feature_extractor.eval()
        classifier.eval()

        trg_pred_labels = np.array([])
        trg_true_labels = np.array([])

        with torch.no_grad():
            for test_data in loader:
                data, label = test_data[0], test_data[1]
                data = data.to(self.device)
                label = label.to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features, domain_type=domain_type)
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                trg_pred_labels = np.append(trg_pred_labels, pred.cpu().numpy())
                trg_true_labels = np.append(trg_true_labels, label.data.cpu().numpy())

        feature_extractor.train()
        classifier.train()

        shared_class = [i for i in self.src_class if i in self.trg_class]
        trg_only_class = [i for i in self.trg_class if i not in self.src_class]
        ind_shared = np.array([lab in shared_class for lab in list(trg_true_labels)])
        ind_trg_only = np.array([lab in trg_only_class for lab in list(trg_true_labels)])

        acc_shared = accuracy_score(np.array(trg_true_labels[ind_shared]).astype(int), np.array(trg_pred_labels[ind_shared]).astype(int))
        acc_trg_only = accuracy_score(np.array(trg_true_labels[ind_trg_only]).astype(int), np.array(trg_pred_labels[ind_trg_only]).astype(int))

        # percentage of samples predicted to be in shared / trg_only classes
        pct_trg_only = 100 * len([label for label in trg_pred_labels if label in trg_only_class]) / len(trg_pred_labels)

        return acc_shared, acc_trg_only, pct_trg_only

    def evaluate(self, ckpt_name='last_checkpoint'):

        ckpt_path = os.path.join(self.home_path, self.scenario_log_dir, ckpt_name + '.pt')
        self.algorithm.load_state_dict(torch.load(ckpt_path)['model_dict'])
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        if not hasattr(self, 'trg_pred_labels'):
            self.trg_pred_labels = {}
        if not hasattr(self, 'trg_true_labels'):
            self.trg_true_labels = {}
        if not hasattr(self, 'trg_loss'):
            self.trg_loss = {}
        self.trg_pred_labels[ckpt_name] = {'train': np.array([]), 'test': np.array([])}
        self.trg_true_labels[ckpt_name] = {'train': np.array([]), 'test': np.array([])}
        self.trg_loss[ckpt_name] = {'train': None, 'test': None}

        evaluate_dl_dict = {'train': self.trg_ul_train_eval_dl, 'test': self.trg_test_dl}
        with torch.no_grad():
            for split, dl in evaluate_dl_dict.items():
                total_loss_ = []
                for dl_data in dl:
                    data, label = dl_data[0], dl_data[1]
                    data = data.to(self.device)
                    label = label.to(self.device)

                    # forward pass
                    features = feature_extractor(data)
                    predictions = classifier(features, domain_type='trg')

                    # compute loss
                    loss = F.cross_entropy(predictions, label)
                    total_loss_.append(loss.item())
                    pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                    self.trg_pred_labels[ckpt_name][split] = np.append(self.trg_pred_labels[ckpt_name][split], pred.cpu().numpy())
                    self.trg_true_labels[ckpt_name][split] = np.append(self.trg_true_labels[ckpt_name][split], label.data.cpu().numpy())
                self.trg_loss[ckpt_name][split] = torch.tensor(total_loss_).mean()

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(self.wandb_tag), hparams_class()

    def load_data(self, src_domain, trg_domain, seed):
        self.class_list = return_classlist(os.path.join(self.dataset_txt_path, 'train_' + src_domain + '.txt'))
        # No test set for src
        self.src_train_dl, self.src_val_dl, _ \
            = data_generator(self.dataset_txt_path, src_domain, self.hparams, True, None, self.im_dir, self.da_setting, self.backbone, seed)
        # trg_train_dl and trg_ul_train_dl is generated using other data_generators according to sampling strategy
        _, self.trg_val_dl, self.trg_test_dl \
            = data_generator(self.dataset_txt_path, trg_domain, self.hparams, False, max(self.num_shots, 1), self.im_dir, self.da_setting, self.backbone, seed)
        if self.sampling in ['kshot', 'random']:
            baseline1shot = baseline1shot_ckpt_path = None
        else:
            # need to load 1-shot Baseline checkpoint
            algorithm_class = get_algorithm_class('Baseline')
            backbone_fe = get_backbone_class(self.backbone)
            baseline1shot_ckpt_path = os.path.join(self.home_path, self.scenario_log_dir, 'best_acc_trg_val_checkpoint.pt')
            baseline1shot_ckpt_path = baseline1shot_ckpt_path.replace(self.da_method, 'Baseline').replace(f'{self.sampling}-{self.num_shots}', 'kshot-1')
            baseline1shot = algorithm_class(len(self.class_list), backbone_fe, self.dataset_configs, self.hparams, self.device, self.src_class, self.trg_class, None)
            baseline1shot = baseline1shot.to(self.device)
        self.trg_train_dl, self.trg_ul_train_dl, self.trg_ul_train_eval_dl = \
            sampling_data_generator(self.dataset_txt_path, trg_domain, self.hparams, self.num_shots, self.im_dir, self.da_setting, self.backbone, self.device,
                algorithm=baseline1shot, ckpt_path=baseline1shot_ckpt_path, sampling_method=self.sampling, seed=seed)


    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self, ckpt_name='last_checkpoint', split='test'):
        '''
        Calculates the acc, f1 and risk values for each seeded run of a cross-domain scenario,
        '''
        acc, macro_f1, weighted_f1 = _calc_metrics(self.trg_pred_labels[ckpt_name], self.trg_true_labels[ckpt_name], self.scenario_log_dir, self.home_path,
            self.class_list, self.src_class, self.trg_class, eval_class='all', ckpt_name=ckpt_name, split=split)
        acc_shared, macro_f1_shared, weighted_f1_shared = _calc_metrics(self.trg_pred_labels[ckpt_name], self.trg_true_labels[ckpt_name], self.scenario_log_dir, self.home_path,
            self.class_list, self.src_class, self.trg_class, eval_class='shared', ckpt_name=ckpt_name, split=split)
        acc_trg_only, macro_f1_trg_only, weighted_f1_trg_only = _calc_metrics(self.trg_pred_labels[ckpt_name], self.trg_true_labels[ckpt_name], self.scenario_log_dir, self.home_path,
            self.class_list, self.src_class, self.trg_class, eval_class='trg_only', ckpt_name=ckpt_name, split=split)
        metrics_colnames = ['acc', 'macro_f1', 'weighted_f1',
            'acc_shared', 'macro_f1_shared', 'weighted_f1_shared',
            'acc_trg_only', 'macro_f1_trg_only', 'weighted_f1_trg_only']
        metrics_values = [acc, macro_f1, weighted_f1,
            acc_shared, macro_f1_shared, weighted_f1_shared,
            acc_trg_only, macro_f1_trg_only, weighted_f1_trg_only]

        run_metrics = {v: metrics_values[k] for k, v in enumerate(metrics_colnames)}
        df = pd.DataFrame(columns=metrics_colnames)
        df.loc[0] = metrics_values

        if split not in self.metrics[ckpt_name].keys():
            self.metrics[ckpt_name][split] = {n: [] for n in metrics_colnames}
        for (key, val) in run_metrics.items():
            self.metrics[ckpt_name][split][key].append(val)

        scores_file_name = f"{ckpt_name}_train_scores.xlsx" if (split == 'train') else f"{ckpt_name}_scores.xlsx"
        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, scores_file_name)
        df.to_excel(scores_save_path, index=False)

    def calc_overall_results(self, ckpt_name='last_checkpoint', split='test'):
        '''
        Calculates results over all seeded runs of a cross-domain scenario, for a single sweep run.
        '''
        exp = self.exp_log_dir

        results = pd.DataFrame(columns=["scenario", "acc", "macro_f1", "weighted_f1",
            "acc_shared", "macro_f1_shared", "weighted_f1_shared",
            "acc_trg_only", "macro_f1_trg_only", "weighted_f1_trg_only"])

        single_exp = os.listdir(exp)
        single_exp = [i for i in single_exp if "_to_" in i]
        single_exp.sort()

        scenarios_ids = np.unique(["_".join(i.split("_")[:3]) for i in single_exp])

        # scenario_run
        scores_file_name = f"{ckpt_name}_train_scores.xlsx" if (split == 'train') else f"{ckpt_name}_scores.xlsx"
        for scenario in single_exp:
            scenario_dir = os.path.join(exp, scenario)
            scores = pd.read_excel(os.path.join(scenario_dir, scores_file_name))
            results = pd.concat([results, scores])
            # Set leftmost column as scenario_run
            results.iloc[len(results) - 1, 0] = scenario

        # Group by scenario
        results = results.loc[:, results.columns != 'scenario'].astype(float)
        avg_results = results.groupby(np.arange(len(results)) // self.num_seeds).mean()
        std_results = results.groupby(np.arange(len(results)) // self.num_seeds).std()

        # Create new row for mean over all scenarios
        avg_results.loc[len(avg_results)] = avg_results.mean()
        # Create leftmost column with label "scenario"
        avg_results.insert(0, "scenario", list(scenarios_ids) + ['mean'], True) 
        std_results.insert(0, "scenario", list(scenarios_ids), True)

        report_file_name_avg = f"{ckpt_name}_train_average_results.xlsx" if (split == 'train') else f"{ckpt_name}_average_results.xlsx"
        report_save_path_avg = os.path.join(exp, report_file_name_avg)
        report_file_name_std = f"{ckpt_name}_train_std_results.xlsx" if (split == 'train') else f"{ckpt_name}_std_results.xlsx"
        report_save_path_std = os.path.join(exp, report_file_name_std)
        avg_results.to_excel(report_save_path_avg)
        std_results.to_excel(report_save_path_std)

        if not hasattr(self, 'averages_results_df'):
            self.averages_results_df = {ckpt_name: {} for ckpt_name in ['last_checkpoint', 'best_acc_src_val_checkpoint', 'best_acc_trg_val_checkpoint']}
        if not hasattr(self, 'std_results_df'):
            self.std_results_df = {ckpt_name: {} for ckpt_name in ['last_checkpoint', 'best_acc_src_val_checkpoint', 'best_acc_trg_val_checkpoint']}
        self.averages_results_df[ckpt_name][split] = avg_results
        self.std_results_df[ckpt_name][split] = std_results
