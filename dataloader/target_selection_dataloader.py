import os
import torch
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

from .dataloader import make_dataset_fromlist, get_data_transforms, Load_Dataset


def kshot(train_im_paths, train_labels, num_shots, seed):
    """
    Generates labeled train samples and unlabeled train samples.
    Used for the target domain.
    """
    count_dict = Counter(train_labels)
    samples_count_dict = defaultdict()
    # if the min number of samples in one class is less than num_shots
    for class_id, count in count_dict.items():
        if count > num_shots:
            samples_count_dict[class_id] = num_shots
        else:
            samples_count_dict[class_id] = count

    # class_id -> list of indices with this class_id, to be sampled from
    samples_indices = {}
    prev = 0
    for class_id in sorted(samples_count_dict.keys()):
        samples_indices[class_id] = list(range(prev, prev + count_dict[class_id]))
        prev += count_dict[class_id]

    selected_ids_dict = {}
    random.seed(seed)
    for class_id in sorted(samples_count_dict.keys()):
        selected_ids_dict[class_id] = random.sample(list(samples_indices[class_id]), samples_count_dict[class_id])

    # select the samples according to the selected random ids
    selected_ids = []
    for class_id in sorted(samples_count_dict.keys()):
        selected_ids.extend([i for i in selected_ids_dict[class_id]])

    return selected_ids


def sampling_data_generator(dataset_txt_path, domain, hparams, num_shots, im_dir, da_setting, backbone, device,
    algorithm=None, ckpt_path=None, sampling_method='kshot', seed=0):
    """
    Generates labeled train dataloader and unlabeled train dataloader.
    Used for the target domain.
    """
    train_txt = os.path.join(dataset_txt_path, 'train_' + domain + '.txt')
    train_im_paths, train_labels = make_dataset_fromlist(train_txt, False, da_setting)

    count_dict = Counter(train_labels)
    num_class = len(count_dict.keys())

    batch_size = hparams['batch_size']
    data_transforms = get_data_transforms(backbone)

    ids = list(range(len(train_im_paths)))
    if sampling_method == 'kshot':
        selected_ids = kshot(train_im_paths, train_labels, num_shots, seed)
    else:
        assert (num_shots > 1)
        # select 1 sample randomly per class
        selected_ids_1shot = kshot(train_im_paths, train_labels, 1, seed)

        # select the rest by sampling method
        num_sample_rest = num_class * (num_shots - 1)
        if sampling_method == 'random':
            random.shuffle(ids)
            ids_rest = [i for i in ids if i not in selected_ids_1shot]
            selected_ids = selected_ids_1shot + ids_rest[:num_sample_rest]
        else:
            raise NotImplementedError

    # select the samples according to the selected random ids
    unselected_ids = [i for i in ids if i not in selected_ids]
    selected_im_paths = [train_im_paths[i] for i in selected_ids]
    selected_labels = [train_labels[i] for i in selected_ids]
    unselected_im_paths = [train_im_paths[i] for i in unselected_ids]
    unselected_labels = [train_labels[i] for i in unselected_ids]

    # Unlabeled
    ul_train_dataset = Load_Dataset(unselected_im_paths, unselected_labels, im_dir, transform=data_transforms['val'], strong_transform=data_transforms['strong'])
    ul_train_dataloader = torch.utils.data.DataLoader(ul_train_dataset, batch_size=batch_size * 2, num_workers=3, shuffle=True, drop_last=True)

    ul_train_eval_dataset = Load_Dataset(unselected_im_paths, unselected_labels, im_dir, transform=data_transforms['test'], test=True)
    ul_train_eval_dataloader = torch.utils.data.DataLoader(ul_train_eval_dataset, batch_size=batch_size, num_workers=3, shuffle=False, drop_last=False)

    # Labeled
    train_dataset = Load_Dataset(selected_im_paths, selected_labels, im_dir, transform=data_transforms['val'],
                                 strong_transform=data_transforms['strong'])
    if len(train_dataset) > 0:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), num_workers=3, shuffle=True, drop_last=True)

    return train_dataloader, ul_train_dataloader, ul_train_eval_dataloader
