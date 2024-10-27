import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import wandb

from models.models import Classifier
from models.loss import CDAC_loss, BCE_softlabels, sigmoid_rampup, CrossEntropyWLogits, AdaMatch_loss, univ_ssda_loss, dst_loss
from utils import weights_init
from torch.cuda.amp import GradScaler


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


class SourceOnly(Algorithm):
    """
    SourceOnly algorithm trained using labeled source data only, with Cross Entropy loss.
    """
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(configs)

        self.feature_extractor = backbone_fe()
        self.classifier = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)
        weights_init(self.classifier)

        params = []
        for key, value in dict(self.feature_extractor.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': hparams['multi'],
                        'weight_decay': hparams['weight_decay']}]
                else:
                    params += [{'params': [value], 'lr': hparams['multi'] * 10,
                        'weight_decay': hparams['weight_decay']}]

        self.optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.optimizer_f = optim.SGD(list(self.classifier.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)

        self.param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            self.param_lr_g.append(param_group["lr"])
        self.param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            self.param_lr_f.append(param_group["lr"])

        self.src_class = src_class
        self.trg_class = trg_class
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.hparams = hparams
        self.device = device

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

        # construct losses for source labeled data
        labeled_features_src = self.feature_extractor(im_data_src)
        predictions_src = self.classifier(labeled_features_src, domain_type='src')

        ce_loss = self.criterion(predictions_src, gt_labels_src)
        ce_loss.backward()
        self.optimizer_g.step()
        self.optimizer_f.step()

        return {'Clf loss': ce_loss}


class Baseline(Algorithm):
    """
    Baseline algorithm trained using labeled source and labeled target data only, with Cross Entropy loss.
    """
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(configs)

        self.feature_extractor = backbone_fe()
        self.classifier = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)
        weights_init(self.classifier)

        params = []
        for key, value in dict(self.feature_extractor.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': hparams['multi'],
                        'weight_decay': hparams['weight_decay']}]
                else:
                    params += [{'params': [value], 'lr': hparams['multi'] * 10,
                        'weight_decay': hparams['weight_decay']}]

        self.optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.optimizer_f = optim.SGD(list(self.classifier.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)

        self.param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            self.param_lr_g.append(param_group["lr"])
        self.param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            self.param_lr_f.append(param_group["lr"])

        self.src_class = src_class
        self.trg_class = trg_class
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.hparams = hparams
        self.device = device

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

        # construct losses for overall labeled data
        num_samp_src = im_data_src.shape[0]
        labeled_features_src = self.feature_extractor(im_data_src)
        predictions_src = self.classifier(labeled_features_src, domain_type='src')
        num_samp_trg = im_data_trg.shape[0]
        labeled_features_trg = self.feature_extractor(im_data_trg)
        predictions_trg = self.classifier(labeled_features_trg, domain_type='trg')
        ce_loss_trg = self.criterion(predictions_trg, gt_labels_trg)
        num_samp = num_samp_src + num_samp_trg
        ce_loss_src = self.criterion(predictions_src, gt_labels_src)
        ce_loss = (num_samp_src / num_samp) * ce_loss_src + (num_samp_trg / num_samp) * ce_loss_trg

        ce_loss.backward()
        self.optimizer_g.step()
        self.optimizer_f.step()

        return {'Clf loss': ce_loss}


class CDAC(Algorithm):
    """
    Cross Domain Adaptive Clustering: https://arxiv.org/abs/2104.09415
    """
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(configs)

        self.feature_extractor = backbone_fe()
        self.classifier = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)
        weights_init(self.classifier)

        params = []
        for key, value in dict(self.feature_extractor.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': hparams['multi'],
                        'weight_decay': hparams['weight_decay']}]
                else:
                    params += [{'params': [value], 'lr': hparams['multi'] * 10,
                        'weight_decay': hparams['weight_decay']}]

        self.optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.optimizer_f = optim.SGD(list(self.classifier.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)

        self.param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            self.param_lr_g.append(param_group["lr"])
        self.param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            self.param_lr_f.append(param_group["lr"])

        self.src_class = src_class
        self.trg_class = trg_class
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.cdac_loss = CDAC_loss()
        self.BCE = BCE_softlabels().to(device)
        self.hparams = hparams
        self.device = device

    def zero_grad_all(self):
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        rampup = sigmoid_rampup(step, self.hparams['rampup_length'])
        w_cons = self.hparams['rampup_coef'] * rampup

        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.zero_grad_all()

        # construct losses for overall labeled data
        num_samp_src = im_data_src.shape[0]
        labeled_features_src = self.feature_extractor(im_data_src)
        predictions_src = self.classifier(labeled_features_src, domain_type='src')
        num_samp_trg = im_data_trg.shape[0]
        labeled_features_trg = self.feature_extractor(im_data_trg)
        predictions_trg = self.classifier(labeled_features_trg, domain_type='trg')
        ce_loss_trg = self.criterion(predictions_trg, gt_labels_trg)
        num_samp = num_samp_src + num_samp_trg
        ce_loss_src = self.criterion(predictions_src, gt_labels_src)
        ce_loss = (num_samp_src / num_samp) * ce_loss_src + (num_samp_trg / num_samp) * ce_loss_trg

        ce_loss.backward(retain_graph=True)
        self.optimizer_g.step()
        self.optimizer_f.step()
        self.zero_grad_all()

        loss, unlabeled_features_trg_ul = self.cdac_loss(self.hparams, self.feature_extractor, self.classifier, im_data_trg_ul, im_data_bar_trg_ul,
            im_data_bar2_trg_ul, self.BCE, w_cons, self.device, None)
        loss.backward()
        # clip norm due to convergence issues
        torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 5)
        self.optimizer_g.step()
        self.optimizer_f.step()

        return {'Clf loss': ce_loss.item(), 'Unlabeled loss': loss.item(), 'Total loss': (ce_loss + loss).item()}


class pretrain(Algorithm):
    """
    Pretraining of PAC: https://arxiv.org/abs/2101.12727
    """
    def __init__(self, backbone_fe, configs, hparams, device, src_class, trg_class):
        super().__init__(configs)

        self.feature_extractor = backbone_fe()
        self.classifier = Classifier(4, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=True)

        params = []
        for key, value in dict(self.feature_extractor.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': hparams['multi'],
                        'weight_decay': hparams['weight_decay']}]
                else:
                    params += [{'params': [value], 'lr': hparams['multi'] * 10,
                        'weight_decay': hparams['weight_decay']}]

        self.scaler = GradScaler()
        self.optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.optimizer_f = optim.SGD(list(self.classifier.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)

        self.param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            self.param_lr_g.append(param_group["lr"])
        self.param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            self.param_lr_f.append(param_group["lr"])

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.hparams = hparams
        self.device = device

    def update(self, im_data_src, im_data_trg, gt_labels_src, gt_labels_trg, step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'],
                                            gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'],
                                            gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        # construct losses for overall labeled data
        num_samp_src = im_data_src.shape[0]
        num_samp_trg = im_data_trg.shape[0]
        num_samp = num_samp_src + num_samp_trg
        labeled_features_src = self.feature_extractor(im_data_src)
        labeled_features_trg = self.feature_extractor(im_data_trg)
        assert not (torch.isnan(labeled_features_trg).any() or torch.isinf(
            labeled_features_trg).any()), "labeled_features_trg contains NaN or Inf values"
        labeled_features_trg_before_classifier = labeled_features_trg.clone()
        predictions_src = self.classifier(labeled_features_src, domain_type='src')
        assert torch.all(labeled_features_trg_before_classifier.eq(
            labeled_features_trg)), "labeled_features_trg has changed after classifier"
        predictions_trg = self.classifier(labeled_features_trg, domain_type='trg')

        ce_loss_src = self.criterion(predictions_src, gt_labels_src)
        ce_loss_trg = self.criterion(predictions_trg, gt_labels_trg)
        ce_loss = (num_samp_src / num_samp) * ce_loss_src + (num_samp_trg / num_samp) * ce_loss_trg

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.scaler.scale(ce_loss).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.step(self.optimizer_f)
        self.scaler.update()
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

        return {'Clf loss': ce_loss.item()}


class PAC(Algorithm):
    """
    Pretraining and Consistency: https://arxiv.org/abs/2101.12727
    """
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(configs)
        self.feature_extractor = backbone_fe()
        if os.path.isfile(backbone_path):
            checkpoint = torch.load(backbone_path)
            print(f'Loading backbone from {backbone_path}')
            self.feature_extractor.load_state_dict(checkpoint['G_state_dict'])
        else:
            raise Exception(
                'Path for backbone {} not found'.format(backbone_path))
        self.classifier = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)

        params = []
        for key, value in dict(self.feature_extractor.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': hparams['multi'],
                        'weight_decay': hparams['weight_decay']}]
                else:
                    params += [{'params': [value], 'lr': hparams['multi'] * 10,
                        'weight_decay': hparams['weight_decay']}]

        self.scaler = GradScaler()
        self.optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.optimizer_f = optim.SGD(list(self.classifier.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)

        self.param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            self.param_lr_g.append(param_group["lr"])
        self.param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            self.param_lr_f.append(param_group["lr"])

        self.src_class = src_class
        self.trg_class = trg_class
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.criterion1 = CrossEntropyWLogits(reduction='none').to(device)
        self.hparams = hparams
        self.device = device

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

        # construct losses for overall labeled data
        labeled_features_src = self.feature_extractor(im_data_src)
        labeled_features_trg = self.feature_extractor(im_data_trg)
        predictions_src = self.classifier(labeled_features_src, domain_type='src')
        predictions_trg = self.classifier(labeled_features_trg, domain_type='trg')
        ce_loss_trg = self.criterion(predictions_trg, gt_labels_trg)
        ce_loss_src = self.criterion(predictions_src, gt_labels_src)
        cls_loss = ce_loss_src + ce_loss_trg
        loss = cls_loss

        if self.hparams.get('cons_wt') and self.hparams['cons_wt'] > 0:
            feats_unl = self.feature_extractor(im_data_trg_ul)
            pls = torch.softmax(self.classifier(feats_unl, domain_type='trg').detach(), dim=1)
            confs, _ = torch.max(pls, dim=1)
            pl_mask = (confs > self.hparams['cons_threshold']).float()
            loss_cons = (self.criterion1(self.classifier(self.feature_extractor(im_data_bar_trg_ul), domain_type='trg'), pls) * pl_mask).mean()
            loss += self.hparams['cons_wt'] * loss_cons

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.step(self.optimizer_f)
        self.scaler.update()
        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

        return {'Clf loss': cls_loss.item(), 'Unlabeled loss': loss_cons.item(), 'Total loss': loss.item()}


class AdaMatch(Algorithm):
    """
    ADAMATCH: A UNIFIED APPROACH TO SEMISUPERVISED LEARNING AND DOMAIN ADAPTATION
    https://arxiv.org/pdf/2106.04732.pdf
    """
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(configs)

        self.feature_extractor = backbone_fe()
        self.classifier = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)
        weights_init(self.classifier)

        params = []
        for key, value in dict(self.feature_extractor.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': hparams['multi'],
                        'weight_decay': hparams['weight_decay']}]
                else:
                    params += [{'params': [value], 'lr': hparams['multi'] * 10,
                        'weight_decay': hparams['weight_decay']}]

        self.optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.optimizer_f = optim.SGD(list(self.classifier.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)

        self.param_lr_g = []
        for param_group in self.optimizer_g.param_groups:
            self.param_lr_g.append(param_group["lr"])
        self.param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            self.param_lr_f.append(param_group["lr"])

        self.src_class = src_class
        self.trg_class = trg_class
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.hparams = hparams
        self.device = device
        self.adamatch_loss = AdaMatch_loss()

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()

        loss, source_loss, target_loss = self.adamatch_loss(self.hparams, self.feature_extractor, self.classifier, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
               gt_labels_src, gt_labels_trg, step, self.hparams['warm_steps'], self.device, self.src_class, self.trg_class)

        loss.backward()
        self.optimizer_g.step()
        self.optimizer_f.step()

        return {'Clf loss': source_loss.item(), 'Unlabeled loss': target_loss.item(), 'Total loss': loss.item(), }


class DST(AdaMatch):
    """
    Debiased Self-Training for Semi-Supervised Learning
    https://arxiv.org/abs/2202.07136
    """
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path, modifications)

        self.classifier2 = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)
        weights_init(self.classifier2)

        self.optimizer_f2 = optim.SGD(list(self.classifier2.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.param_lr_f2 = []
        for param_group in self.optimizer_f2.param_groups:
            self.param_lr_f2.append(param_group["lr"])

        self.dst_loss = dst_loss()

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f2 = inv_lr_scheduler(self.param_lr_f2, self.optimizer_f2, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.optimizer_f2.zero_grad()

        loss1, loss2 = self.dst_loss(self.hparams, self.feature_extractor, self.classifier, self.classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
               gt_labels_src, gt_labels_trg, step, self.hparams['warm_steps'], self.device, self.src_class, self.trg_class)
        total_loss = loss1 + loss2
        total_loss.backward()
        self.optimizer_g.step()
        self.optimizer_f.step()
        self.optimizer_f2.step()

        return {'Clf loss': loss1.item(), 'Clf loss2': loss2.item(), 'Total loss': total_loss.item()}


# Proposed method

class Proposed(AdaMatch):
    def __init__(self, num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path=None, modifications=None):
        super().__init__(num_class, backbone_fe, configs, hparams, device, src_class, trg_class, backbone_path, modifications)

        self.classifier2 = Classifier(num_class, src_class, trg_class, inc=self.feature_extractor.output_col, temp=hparams['temp'], pretrain=False)
        weights_init(self.classifier2)

        self.optimizer_f2 = optim.SGD(list(self.classifier2.parameters()), lr=hparams['lr_f'], momentum=0.9, weight_decay=hparams['weight_decay'], nesterov=True)
        self.param_lr_f2 = []
        for param_group in self.optimizer_f2.param_groups:
            self.param_lr_f2.append(param_group["lr"])

        self.proposed_loss = univ_ssda_loss()

    def update(self, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul, im_data_bar2_trg_ul,
            gt_labels_src, gt_labels_trg, step, total_step):
        self.optimizer_g = inv_lr_scheduler(self.param_lr_g, self.optimizer_g, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f = inv_lr_scheduler(self.param_lr_f, self.optimizer_f, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])
        self.optimizer_f2 = inv_lr_scheduler(self.param_lr_f2, self.optimizer_f2, step, init_lr=self.hparams['lr'], gamma=self.hparams['gamma'])

        self.optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        self.optimizer_f2.zero_grad()

        loss, labeled_loss, unlabeled_loss, loss2 = self.proposed_loss(self.hparams, self.feature_extractor, self.classifier, self.classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
               gt_labels_src, gt_labels_trg, step, self.hparams['warm_steps'], self.device, self.src_class, self.trg_class)
        total_loss = loss + loss2
        total_loss.backward()
        self.optimizer_g.step()
        self.optimizer_f.step()
        self.optimizer_f2.step()

        return {'Clf loss': labeled_loss.item(), 'Unlabeled loss': unlabeled_loss.item(), 'Clf loss2': loss2.item(), 'Total loss': total_loss.item()}