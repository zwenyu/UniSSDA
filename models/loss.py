import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CDAC_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hparams, backbone_fe, classifier, im_data, im_data_bar, im_data_bar2, BCE, w_cons, device, target):
        aac_loss, pl_loss, con_loss, feat = self.get_losses_unlabeled(hparams, backbone_fe, classifier, im_data, im_data_bar, im_data_bar2, BCE, w_cons,
            device, target)

        return aac_loss + pl_loss + con_loss, feat

    def get_losses_unlabeled(self, hparams, G, F1, im_data, im_data_bar, im_data_bar2, BCE, w_cons, device, target):
        """ Get losses for unlabeled samples."""
        feat = G(im_data)
        feat_bar = G(im_data_bar)
        feat_bar2 = G(im_data_bar2)

        output = F1(feat, reverse=True, eta=1.0, domain_type='trg')
        output_bar = F1(feat_bar, reverse=True, eta=1.0, domain_type='trg')
        prob, prob_bar = F.softmax(output, dim=1), F.softmax(output_bar, dim=1)

        # loss for adversarial adaptive clustering
        aac_loss = self.advbce_unlabeled(hparams, target=target, feat=feat, prob=prob, prob_bar=prob_bar, device=device, bce=BCE)

        output = F1(feat, domain_type='trg')
        output_bar = F1(feat_bar, domain_type='trg')
        output_bar2 = F1(feat_bar2, domain_type='trg')

        prob = F.softmax(output, dim=1)
        prob_bar = F.softmax(output_bar, dim=1)
        prob_bar2 = F.softmax(output_bar2, dim=1)

        max_probs, pseudo_labels = torch.max(prob.detach_(), dim=-1)
        mask = max_probs.ge(hparams['threshold']).float()

        # loss for pseudo labeling
        pl_loss = (F.cross_entropy(output_bar2, pseudo_labels, reduction='none') * mask).mean()

        # loss for consistency
        con_loss = w_cons * F.mse_loss(prob_bar, prob_bar2)

        return aac_loss, pl_loss, con_loss, feat

    def advbce_unlabeled(self, hparams, target, feat, prob, prob_bar, device, bce):
        """ Construct adversarial adaptive clustering loss."""
        target_ulb = self.pairwise_target(hparams, feat, target, device)
        prob_bottleneck_row, _ = self.PairEnum2D(prob)
        _, prob_bottleneck_col = self.PairEnum2D(prob_bar)
        adv_bce_loss = -bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
        return adv_bce_loss

    def pairwise_target(self, hparams, feat, target, device):
        """ Produce pairwise similarity label."""
        feat_detach = feat.detach()
        # For unlabeled data
        if target is None:
            rank_feat = feat_detach
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = self.PairEnum2D(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :hparams['topk']], rank_idx2[:, :hparams['topk']]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = 0
        # For labeled data
        elif target is not None:
            target_row, target_col = self.PairEnum1D(target)
            target_ulb = torch.zeros(target.size(0) * target.size(0)).float().to(device)
            target_ulb[target_row == target_col] = 1
        else:
            raise ValueError('Please check your target.')
        return target_ulb

    def PairEnum1D(self, x):
        """ Enumerate all pairs of feature in x with 1 dimension."""
        assert x.ndimension() == 1, 'Input dimension must be 1'
        x1 = x.repeat(x.size(0), )
        x2 = x.repeat(x.size(0)).view(-1,x.size(0)).transpose(1, 0).reshape(-1)
        return x1, x2

    def PairEnum2D(self, x):
        """ Enumerate all pairs of feature in x with 2 dimensions."""
        assert x.ndimension() == 2, 'Input dimension must be 2'
        x1 = x.repeat(x.size(0), 1)
        x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
        return x1, x2


class BCE(nn.Module):
    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


class BCE_softlabels(nn.Module):
    """ Construct binary cross-entropy loss."""
    eps = 1e-7

    def forward(self, prob1, prob2, simi):
        P = prob1 * prob2
        P = P.sum(1)
        neglogP = - (simi * torch.log(P + BCE.eps) + (1. - simi) * torch.log(1. - P + BCE.eps))
        return neglogP.mean()


class CrossEntropyWLogits(torch.nn.Module):
    def __init__(self, reduction='mean'):
        # can support different kinds of reductions if needed
        super(CrossEntropyWLogits, self).__init__()
        assert reduction == 'mean' or reduction == 'none', \
            'utils.loss.CrossEntropyWLogits: reduction not recognized'
        self.reduction = reduction

    def forward(self, logits, targets):
        # shape of targets needs to match that of preds
        log_preds = torch.log_softmax(logits, dim=1)
        if self.reduction == 'mean':
            return torch.mean(torch.sum(-targets*log_preds, dim=1), dim=0)
        else:
            return torch.sum(-targets*log_preds, dim=1)


class AdaMatch_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hparams, backbone_fe, classifier, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class=None, trg_class=None):
        loss, source_loss, target_loss = self.get_losses(hparams, backbone_fe, classifier, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class, trg_class)
        return loss, source_loss, target_loss

    def get_losses(self, hparams, feature_extractor, classifier, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class, trg_class):
        # in this function, source refers to labeled samples, target refers to unlabeled samples
        all_labeled_img = torch.cat([im_data_src, im_data_trg])
        all_labeled_bar_img = torch.cat([im_data_bar_src, im_data_bar_trg])
        labels_source = torch.cat([gt_labels_src, gt_labels_trg])

        data_combined_img = torch.cat([all_labeled_img, all_labeled_bar_img, im_data_trg_ul, im_data_bar_trg_ul], dim=0)
        source_combined_img = torch.cat([all_labeled_img, all_labeled_bar_img], dim=0)
        data_combined = feature_extractor(data_combined_img)
        source_total = source_combined_img.size(0)

        # Generate two different outputs of source input
        logits_combined = classifier(data_combined, domain_type='all')
        logits_source_p = logits_combined[:source_total]

        self._disable_batchnorm_tracking(feature_extractor)
        self._disable_batchnorm_tracking(classifier)
        source_combined = feature_extractor(source_combined_img)
        logits_source_pp = classifier(source_combined, domain_type='all')
        self._enable_batchnorm_tracking(feature_extractor)
        self._enable_batchnorm_tracking(classifier)

        lamb = torch.rand_like(logits_source_p).to(device)
        final_logits_source = (lamb * logits_source_p) + ((1 - lamb) * logits_source_pp)

        logits_source_weak = final_logits_source[:all_labeled_img.size(0)]
        pseudolabels_source = F.softmax(logits_source_weak, dim=1)

        # softmax for logits of weakly augmented target images
        logits_target = logits_combined[source_total:]
        logits_target_weak = logits_target[:im_data_trg_ul.size(0)]
        pseudolabels_target = F.softmax(logits_target_weak, dim=1)

        # align target label distribtion to source label distribution
        expectation_ratio = (1e-6 + torch.mean(pseudolabels_source, dim=0)) / (1e-6 + torch.mean(pseudolabels_target, dim=0))
        final_pseudolabels = F.normalize((pseudolabels_target * expectation_ratio), p=1, dim=1)  # L1 normalization

        # perform relative confidence thresholding
        row_wise_max, _ = torch.max(pseudolabels_source, dim=1)
        final_sum = torch.mean(row_wise_max, 0)

        # define relative confidence threshold
        c_tau = hparams['tau'] * final_sum

        max_values, final_pseudolabels_cls = torch.max(final_pseudolabels, dim=1)
        mask = (max_values >= c_tau).float()

        # compute loss
        source_loss = self._compute_source_loss(logits_source_weak, final_logits_source[all_labeled_img.size(0):],
                                                labels_source)
        pseudolabels = final_pseudolabels_cls.detach()
        target_loss = self._compute_target_loss(pseudolabels, logits_target[im_data_trg_ul.size(0):], mask)

        # compute target loss weight (mu)
        pi = torch.tensor(np.pi, dtype=torch.float).to(device)
        step = torch.tensor(step, dtype=torch.float).to(device)
        mu = 0.5 - torch.cos(torch.minimum(pi, (pi * step) / (warm_steps + 1e-5))) / 2

        # get total loss
        loss = source_loss + (mu * target_loss)

        return loss, source_loss, target_loss

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.CrossEntropyLoss()  # default: `reduction="mean"`
        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)

        # return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.CrossEntropyLoss(reduction="none")

        loss = loss_function(logits_strong, pseudolabels)

        return (loss * mask).mean()

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)


class dst_loss(AdaMatch_loss):
    def __init__(self):
        super().__init__()

    def forward(self, hparams, backbone_fe, classifier, classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class=None, trg_class=None, ablation=''):
        loss1, loss2 = self.get_losses(hparams, backbone_fe, classifier, classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class, trg_class, ablation)
        return loss1, loss2

    def get_losses(self, hparams, feature_extractor, classifier, classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class, trg_class, ablation):
        # in this function, source refers to labeled samples, target refers to unlabeled samples
        all_labeled_img = torch.cat([im_data_src, im_data_trg])
        all_labeled_bar_img = torch.cat([im_data_bar_src, im_data_bar_trg])
        labels_source = torch.cat([gt_labels_src, gt_labels_trg])

        data_combined_img = torch.cat([all_labeled_img, all_labeled_bar_img, im_data_trg_ul, im_data_bar_trg_ul], dim=0)
        source_combined_img = torch.cat([all_labeled_img, all_labeled_bar_img], dim=0)
        data_combined = feature_extractor(data_combined_img)
        source_total = source_combined_img.size(0)

        ### first classifier is trained only on labeled samples
        # Generate two different outputs of source input
        logits_combined = classifier(data_combined, domain_type='all')
        logits_source_p = logits_combined[:source_total]
        logits_target = logits_combined[source_total:]

        if 'nologit' not in ablation:
            self._disable_batchnorm_tracking(feature_extractor)
            self._disable_batchnorm_tracking(classifier)
            source_combined = feature_extractor(source_combined_img)
            logits_source_pp = classifier(source_combined, domain_type='all')
            self._enable_batchnorm_tracking(feature_extractor)
            self._enable_batchnorm_tracking(classifier)

            lamb = torch.rand_like(logits_source_p).to(device)
            final_logits_source = (lamb * logits_source_p) + ((1 - lamb) * logits_source_pp)
        else:
            final_logits_source = logits_source_p

        logits_source_weak = final_logits_source[:all_labeled_img.size(0)]
        logits_source_strong = final_logits_source[all_labeled_img.size(0):]

        # compute loss
        loss1 = self._compute_source_loss(logits_source_weak, logits_source_strong, labels_source)

        ### compute target pseudolabels from first classifier
        pseudolabels_source = F.softmax(logits_source_weak, dim=1)
        pseudolabels_target = F.softmax(logits_target, dim=1)
        final_pseudolabels = pseudolabels_target[:im_data_trg_ul.size(0)]

        shared_class = [i for i in src_class if i in trg_class]
        src_only_class = [i for i in src_class if i not in trg_class]
        trg_only_class = [i for i in trg_class if i not in src_class]

        # set src_only_class classes to zero
        final_pseudolabels[:, src_only_class] = 0.
        final_pseudolabels = F.normalize(final_pseudolabels, p=1, dim=1)

        # modify: perform relative confidence thresholding based on labeled distribution
        pseudolabels_thresh = pseudolabels_source.detach()
        row_wise_max, _ = torch.max(pseudolabels_thresh, dim=1)
        final_sum = torch.mean(row_wise_max, 0)

        # define relative confidence threshold
        c_tau = hparams['tau'] * final_sum

        max_values, final_pseudolabels_cls = torch.max(final_pseudolabels, dim=1)
        mask = (max_values >= c_tau).float()
        mask[mask == 0] = (len(trg_only_class) / (len(trg_only_class) + len(shared_class)))
        pseudolabels = final_pseudolabels_cls.detach()

        ### second classifier is trained on labeled and pseudolabeled samples
        logits2_source = classifier2(data_combined[:source_total], domain_type='all')
        logits2_target = classifier2(data_combined[source_total:], domain_type='all')
        loss2_source = self._compute_source_loss(logits2_source[:all_labeled_img.size(0)],
            logits2_source[all_labeled_img.size(0):], labels_source)

        loss2_target = self._compute_target_loss(pseudolabels, logits2_target[im_data_trg_ul.size(0):], mask)

        # compute target loss weight (mu)
        pi = torch.tensor(np.pi, dtype=torch.float).to(device)
        step = torch.tensor(step, dtype=torch.float).to(device)
        mu = 0.5 - torch.cos(torch.minimum(pi, (pi * step) / (warm_steps + 1e-5))) / 2

        # get total loss
        loss2 = loss2_source + (mu * loss2_target)

        return loss1, loss2


class univ_ssda_loss(AdaMatch_loss):
    def __init__(self):
        super().__init__()

    def forward(self, hparams, backbone_fe, classifier, classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class=None, trg_class=None, ablation=''):
        loss, source_loss, target_loss, loss2 = self.get_losses(hparams, backbone_fe, classifier, classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class, trg_class, ablation)
        return loss, source_loss, target_loss, loss2

    def get_losses(self, hparams, feature_extractor, classifier, classifier2, im_data_src, im_data_bar_src, im_data_trg, im_data_bar_trg, im_data_trg_ul, im_data_bar_trg_ul,
            gt_labels_src, gt_labels_trg, step, warm_steps, device, src_class, trg_class, ablation):
        # in this function, source refers to labeled samples, target refers to unlabeled samples
        all_labeled_img = torch.cat([im_data_src, im_data_trg])
        all_labeled_bar_img = torch.cat([im_data_bar_src, im_data_bar_trg])
        labels_source = torch.cat([gt_labels_src, gt_labels_trg])

        data_combined_img = torch.cat([all_labeled_img, all_labeled_bar_img, im_data_trg_ul, im_data_bar_trg_ul], dim=0)
        source_combined_img = torch.cat([all_labeled_img, all_labeled_bar_img], dim=0)
        data_combined = feature_extractor(data_combined_img)
        source_total = source_combined_img.size(0)

        # second classifier is trained only on labeled samples
        self._disable_batchnorm_tracking(feature_extractor)
        with torch.no_grad():
            labeled_features_src = feature_extractor(im_data_src).detach()
            labeled_features_trg = feature_extractor(im_data_trg).detach()
        self._enable_batchnorm_tracking(feature_extractor)
        logits2_src = classifier2(labeled_features_src, domain_type='src')
        logits2_trg = classifier2(labeled_features_trg, domain_type='trg')

        # Generate two different outputs of source input
        logits_combined = classifier(data_combined, domain_type='all')
        logits_source_p = logits_combined[:source_total]

        if 'nologit' not in ablation:
            self._disable_batchnorm_tracking(feature_extractor)
            self._disable_batchnorm_tracking(classifier)
            source_combined = feature_extractor(source_combined_img)
            logits_source_pp = classifier(source_combined, domain_type='all')
            self._enable_batchnorm_tracking(feature_extractor)
            self._enable_batchnorm_tracking(classifier)

            lamb = torch.rand_like(logits_source_p).to(device)
            final_logits_source = (lamb * logits_source_p) + ((1 - lamb) * logits_source_pp)
        else:
            final_logits_source = logits_source_p

        logits_source_weak = final_logits_source[:all_labeled_img.size(0)]
        logits_source_strong = final_logits_source[all_labeled_img.size(0):]
        pseudolabels_source = F.softmax(logits_source_weak, dim=1)

        # softmax for logits of weakly augmented target images
        logits_target = logits_combined[source_total:]
        logits_target_weak = logits_target[:im_data_trg_ul.size(0)]
        pseudolabels_target = F.softmax(logits_target_weak, dim=1)

        # modify: align porportion of shared and private classes in target domain unlabeled distribution according to classifier2
        self._disable_batchnorm_tracking(feature_extractor)
        with torch.no_grad():
            unlabeled_features_trg = feature_extractor(im_data_trg_ul).detach()
        self._enable_batchnorm_tracking(feature_extractor)
        logits2_target_weak = classifier2(unlabeled_features_trg, domain_type='trg')
        pseudolabels2_target = F.softmax(logits2_target_weak, dim=1)

        shared_class = [i for i in src_class if i in trg_class]
        src_only_class = [i for i in src_class if i not in trg_class]
        trg_only_class = [i for i in trg_class if i not in src_class]
        if 'nogrpl' not in ablation:
            est_prop_shared = pseudolabels_target[:, shared_class].sum(dim=1, keepdim=True).detach()
            est_prop_src_only = pseudolabels_target[:, src_only_class].sum(dim=1, keepdim=True).detach()
            est_prop_trg_only = pseudolabels_target[:, trg_only_class].sum(dim=1, keepdim=True).detach()
            est2_prop_shared = pseudolabels2_target[:, shared_class].sum(dim=1, keepdim=True).detach()
            est2_prop_src_only = pseudolabels2_target[:, src_only_class].sum(dim=1, keepdim=True).detach()
            est2_prop_trg_only = pseudolabels2_target[:, trg_only_class].sum(dim=1, keepdim=True).detach()

            pseudolabels_target[:, shared_class] = pseudolabels_target[:, shared_class] * est2_prop_shared / (1e-6 + est_prop_shared)
            pseudolabels_target[:, src_only_class] = pseudolabels_target[:, src_only_class] * est2_prop_src_only / (1e-6 + est_prop_src_only)
            pseudolabels_target[:, trg_only_class] = pseudolabels_target[:, trg_only_class] * est2_prop_trg_only / (1e-6 + est_prop_trg_only)
            final_pseudolabels = F.normalize(pseudolabels_target, p=1, dim=1)  # L1 normalization
        else:
            final_pseudolabels = pseudolabels_target
        if 'nopredavg' not in ablation:
            final_pseudolabels = (final_pseudolabels + pseudolabels2_target) / 2
        # set src_only_class classes to zero
        final_pseudolabels[:, src_only_class] = 0.
        final_pseudolabels = F.normalize(final_pseudolabels, p=1, dim=1)

        # modify: perform relative confidence thresholding based on labeled distribution
        pseudolabels_thresh = pseudolabels_source.detach()
        row_wise_max, _ = torch.max(pseudolabels_thresh, dim=1)
        final_sum = torch.mean(row_wise_max, 0)

        # define relative confidence threshold
        c_tau = hparams['tau'] * final_sum

        max_values, final_pseudolabels_cls = torch.max(final_pseudolabels, dim=1)
        mask = (max_values >= c_tau).float()
        mask[mask == 0] = (len(trg_only_class) / (len(trg_only_class) + len(shared_class)))

        # compute loss
        source_loss = self._compute_source_loss(logits_source_weak, logits_source_strong,
                                                labels_source)
        pseudolabels = final_pseudolabels_cls.detach()
        target_loss = self._compute_target_loss(pseudolabels, logits_target[im_data_trg_ul.size(0):], mask)

        # compute target loss weight (mu)
        pi = torch.tensor(np.pi, dtype=torch.float).to(device)
        step = torch.tensor(step, dtype=torch.float).to(device)
        mu = 0.5 - torch.cos(torch.minimum(pi, (pi * step) / (warm_steps + 1e-5))) / 2

        # get total loss
        loss = source_loss + (mu * target_loss)

        # classifier2 loss
        num_samp_src = im_data_src.shape[0]
        num_samp_trg = im_data_trg.shape[0]
        num_samp = num_samp_src + num_samp_trg
        loss2_src = F.cross_entropy(logits2_src, gt_labels_src)  # mask this with mask_labeled_src is slightly worse
        loss2_trg = F.cross_entropy(logits2_trg, gt_labels_trg)
        loss2 = (num_samp_src / num_samp) * loss2_src + (num_samp_trg / num_samp) * loss2_trg

        return loss, source_loss, target_loss, loss2


def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))