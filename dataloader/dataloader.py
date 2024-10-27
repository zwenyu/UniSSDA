import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .randaugment import RandAugmentMC
import random
from collections import Counter, defaultdict
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset

import os
import numpy as np

I2T = transforms.ToTensor()


########## data transformation #############################

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def get_data_transforms(backbone):
    if backbone == 'AlexNetBase':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'strong': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

########## data loader #############################

class Load_Dataset(Dataset):
    def __init__(self, im_paths, labels, im_dir, transform=None, strong_transform=None, target_transform=None, rot=False,test=None):
        super().__init__()
        self.im_paths = im_paths
        self.labels = labels
        self.transform = transform
        self.strong_transform = strong_transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.im_dir = im_dir
        self.test = test
        self.rot = rot

    def __getitem__(self, index):
        path = os.path.join(self.im_dir, self.im_paths[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.strong_transform is not None:
            img_bar = self.strong_transform(img)
            img_bar2 = self.strong_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None and not self.rot:
            img = self.transform(img)
        if self.rot:
            all_rotated_imgs = [
                self.transform(TF.rotate(img, -90)),
                self.transform(img),
                self.transform(TF.rotate(img, 90)),
                self.transform(TF.rotate(img, 180))]
            #print(self.transform)
            all_rotated_imgs = torch.stack(all_rotated_imgs, dim=0)
            target = torch.tensor(len(all_rotated_imgs) * [target])
            rot_target = torch.LongTensor([0, 1, 2, 3])
            assert not torch.isnan(all_rotated_imgs).any(), f"NaN values found in img after transform at index {index}"
            assert not torch.isinf(all_rotated_imgs).any(), f"Inf values found in img after transform at index {index}"
            return all_rotated_imgs, target, rot_target, index
        if not self.test:
            return img, target, index, img_bar, img_bar2
        else:
            return img, target, self.im_paths[index], index

    def __len__(self):
        return len(self.im_paths)


def data_generator(dataset_txt_path, domain, hparams, is_source, num_shots, im_dir, da_setting, backbone, seed):
    """
    Generates train, validation and test dataloaders for the given domain_id of the dataset.
    Different transforms will be applied on the train dataset depending on whether the domain is taken as source.
    """
    train_txt = os.path.join(dataset_txt_path, 'train_' + domain + '.txt')
    train_im_paths, train_labels = make_dataset_fromlist(train_txt, is_source, da_setting)
    #print(f'loading src train data from: {train_txt}')

    validation_txt = os.path.join(dataset_txt_path, 'val_' + domain + '.txt')
    validation_im_paths, validation_labels = make_dataset_fromlist(validation_txt, is_source, da_setting)

    test_txt = os.path.join(dataset_txt_path, 'test_' + domain + '.txt')
    test_im_paths, test_labels = make_dataset_fromlist(test_txt, is_source, da_setting)

    batch_size = hparams['batch_size']
    data_transforms = get_data_transforms(backbone)

    train_dataset = Load_Dataset(train_im_paths, train_labels, im_dir, transform=data_transforms['train'],
                                    strong_transform=data_transforms['strong'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), num_workers=3,
                                                    shuffle=True, drop_last=True)

    if (not is_source) and (num_shots is not None):
        count_dict = Counter(validation_labels)
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

        selected_ids = {}
        random.seed(seed)
        for class_id in sorted(samples_count_dict.keys()):
            selected_ids[class_id] = random.sample(list(samples_indices[class_id]), samples_count_dict[class_id])

        # select the samples according to the selected random ids
        validation_selected_im_paths = []
        validation_selected_labels = []

        for class_id in sorted(samples_count_dict.keys()):
            validation_selected_im_paths.extend([validation_im_paths[i] for i in selected_ids[class_id]])
            validation_selected_labels.extend([validation_labels[i] for i in selected_ids[class_id]])
    else:
        validation_selected_im_paths = validation_im_paths
        validation_selected_labels = validation_labels
    validation_dataset = Load_Dataset(validation_selected_im_paths, validation_selected_labels, im_dir, transform=data_transforms['val'],
                                      strong_transform=data_transforms['strong'])
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=3,
                                                        shuffle=False, drop_last=False)
    test_dataset = Load_Dataset(test_im_paths, test_labels, im_dir, transform=data_transforms['test'], test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=3, shuffle=False,
                                                  drop_last=False)

    return train_dataloader, validation_dataloader, test_dataloader


def pretrain_data_generator(dataset_txt_path, domain, hparams, im_dir, is_source, da_setting,backbone):

    train_txt = os.path.join(dataset_txt_path, 'train_' + domain + '.txt')
    #print('make_dataset_fromlist Train...')
    train_im_paths, train_labels = make_dataset_fromlist(train_txt, is_source, da_setting)
    val_txt = os.path.join(dataset_txt_path, 'val_' + domain + '.txt')
    #print('make_dataset_fromlist Val...')
    val_im_paths, val_labels = make_dataset_fromlist(val_txt, is_source, da_setting)
    data_transforms = get_data_transforms(backbone)
    batch_size = hparams['batch_size']
    train_dataset = Load_Dataset(train_im_paths, train_labels, im_dir, transform=data_transforms['strong'], rot=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), num_workers=3,
                                                   shuffle=True, drop_last=True)
    val_dataset = Load_Dataset(val_im_paths, val_labels, im_dir, transform=data_transforms['strong'], rot=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), num_workers=3,
                                                   shuffle=True, drop_last=True)
    return train_dataloader,val_dataloader


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def return_classlist(dataset_txt_file):
    with open(dataset_txt_file) as f:
        label_list = []
        for x in f.readlines():
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


def make_dataset_fromlist(dataset_txt_file, is_source, da_setting='closed'):
    # print("image_list", image_list)
    select_cat = get_select_class(dataset_txt_file,is_source,da_setting)
    #print(f'Len of select labels {len(select_cat)}')
    im_paths = []
    labels = []
    with open(dataset_txt_file) as f:
        lines = f.read().splitlines()
        for line in lines:
            im_path, label = line.split(' ')
            if int(label) in select_cat:
                im_paths.append(im_path)
                labels.append(int(label))
    return im_paths, labels


def get_select_class(dataset_txt_file, is_source, da_setting='closed'):
    """
    Class splits for closed, partial, open, open-partial set DA
    -   Partial
        - Office-31: set all categories as source classes, 21-31 categories as target classes
        - Office-Home: set all categories as source classes, 41-65 categories as target classes
        - DomainNet: set all categories as source classes, 81-126 categories as target classes
    -   Open
        - Office-31: set first 20 categories as shared, 21-31 categories belong only to target
        - Office-Home: set first 40 categories as shared, 41-65 categories belong only to target
        - DomainNet: set first 80 categories as shared, 81-126 categories belong only to target
    -   Open-partial
        - Office-31: set first 10 categories as shared, 11-20 categories belong only to source, 21-31 categories belong only to target
        - Office-Home: set first 20 categories as shared, 21-40 categories belong only to source, 41-65 categories belong only to target
        - Office-Home: set first 40 categories as shared, 41-80 categories belong only to source, 81-126 categories belong only to target
    """
    if da_setting == 'partial':
        if 'office31' in dataset_txt_file:
            select_cat = list(range(31)) if is_source else list(range(20, 31))
        elif 'office_home' in dataset_txt_file:
            select_cat = list(range(65)) if is_source else list(range(40, 65))
        elif 'domain_net' in dataset_txt_file:
            select_cat = list(range(126)) if is_source else list(range(80, 126))
    elif da_setting == 'open':
        if 'office31' in dataset_txt_file:
            select_cat = list(range(20)) if is_source else list(range(31))
        elif 'office_home' in dataset_txt_file:
            select_cat = list(range(40)) if is_source else list(range(65))
        elif 'domain_net' in dataset_txt_file:
            select_cat = list(range(80)) if is_source else list(range(126))
    elif da_setting == 'openpartial':
        if 'office31' in dataset_txt_file:
            select_cat = list(range(20)) if is_source else list(range(10)) + list(range(20, 31))
        elif 'office_home' in dataset_txt_file:
            select_cat = list(range(40)) if is_source else list(range(20)) + list(range(40, 65))
        elif 'domain_net' in dataset_txt_file:
            select_cat = list(range(80)) if is_source else list(range(40)) + list(range(80, 126))
    elif da_setting == 'closed':
        if 'office31' in dataset_txt_file:
            select_cat = list(range(31))
        elif 'office_home' in dataset_txt_file:
            select_cat = list(range(65))
        elif 'domain_net' in dataset_txt_file:
            select_cat = list(range(126))
    else:
        assert ('domain_net' in dataset_txt_file)
        # further analysis on domain_net
        # increasing number of source private classes
        if da_setting == 'openpartial_src_pvt_0':
            select_cat = list(range(40)) if is_source else list(range(40)) + list(range(80, 126))
        if da_setting == 'openpartial_src_pvt_10':
            select_cat = list(range(50)) if is_source else list(range(40)) + list(range(80, 126))
        if da_setting == 'openpartial_src_pvt_20':
            select_cat = list(range(60)) if is_source else list(range(40)) + list(range(80, 126))
        if da_setting == 'openpartial_src_pvt_30':
            select_cat = list(range(70)) if is_source else list(range(40)) + list(range(80, 126))
        # increasing number of target private classses
        if da_setting == 'openpartial_trg_pvt_0':
            select_cat = list(range(80)) if is_source else list(range(40))
        if da_setting == 'openpartial_trg_pvt_10':
            select_cat = list(range(80)) if is_source else list(range(40)) + list(range(80, 90))
        if da_setting == 'openpartial_trg_pvt_20':
            select_cat = list(range(80)) if is_source else list(range(40)) + list(range(80, 100))
        if da_setting == 'openpartial_trg_pvt_30':
            select_cat = list(range(80)) if is_source else list(range(40)) + list(range(80, 110))
        if da_setting == 'openpartial_trg_pvt_40':
            select_cat = list(range(80)) if is_source else list(range(40)) + list(range(80, 120))

    return select_cat