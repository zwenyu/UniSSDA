"""
Ensures that datasets are of same as in as txt folder.
Each domain of a dataset has its original class distribution, then generates text files
of which each line is in the form 'relative_path_to_images_folder class_id' for each split,
50% train, 20% validation, 30% test.
This script will work for all datasets with the file structure 'dataset/domain/class/image', as in office_home.
"""

import os
import argparse
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='office_home', type=str)
parser.add_argument('--data_root', default='..', type=str)

args = parser.parse_args()

if __name__ == "__main__":

    cwd = Path.cwd()
    data_dir = os.path.join(args.data_root, args.dataset)
    txt_dir = os.path.join(cwd, args.dataset)
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    domains = sorted([f.name for f in os.scandir(data_dir) if f.is_dir()])

    if args.dataset == 'office_home':
        class_list = sorted([f.name for f in os.scandir(os.path.join(data_dir, domains[0]))])
    elif args.dataset == 'domain_net':
        with open('../domain_net_class_list.txt') as f:
            class_list = f.readlines()
        class_list = sorted([line.rstrip('\n') for line in class_list])

    ### dataset size in txt folder
    # class -> min_samples across all domains
    min_samples_dict = {}

    # For each class, across all domains, make sure the number of samples is the same to
    # ensure the same class distribution across all domains.
    # Select only min_samples samples to retain for this class in all domains
    # When split, splits will be the same size across all domains
    sample_size_dict = {d: {} for d in domains}
    for c in class_list:
        min_samples = float('inf')
        for d in domains:
            num_samples = len(list(os.scandir(os.path.join(data_dir, d, c))))
            sample_size_dict[d][c] = num_samples
            if num_samples < min_samples:
                min_samples = num_samples
        min_samples_dict[c] = min_samples
    dataset_size = sum(min_samples_dict.values())

    # Change directory to write txt files
    os.chdir(txt_dir)

    # Carry out the splits for each domain using stratified sampling
    for d in domains:
        paths = []
        class_ids = []
        for class_id, c in enumerate(class_list):
            retained_num_samples = round((sample_size_dict[d][c] / sum(sample_size_dict[d].values())) * dataset_size)
            retained_samples_paths = [os.path.relpath(f.path, data_dir) for f in os.scandir(os.path.join(data_dir, d, c))][:retained_num_samples]
            paths.extend(retained_samples_paths)
            class_ids.extend([class_id] * retained_num_samples)

        # 50% train, 20% validation, 30% test
        # First split into 50% train, 50% (validation and test)
        X_train, X_val_test, y_train, y_val_test = train_test_split(paths, class_ids, stratify=class_ids, test_size=0.5, random_state=1)

        # Second split of 50% (validation and test) into 20% validation and 30% test
        # test_size * 50% (validation and test) = 30% of entire set, test_size = 0.6
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, stratify=y_val_test, test_size=0.6, random_state=1)

        with open(f"train_{d}.txt", "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(sorted(zip(X_train, y_train), key=lambda x: x[1]))

        with open(f"val_{d}.txt", "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(sorted(zip(X_val, y_val), key=lambda x: x[1]))

        with open(f"test_{d}.txt", "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(sorted(zip(X_test, y_test), key=lambda x: x[1]))
