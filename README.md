This repository contains code demonstrating the UniSSDA method in our CVPR 2024 paper [Universal Semi-Supervised Domain Adaptation by Mitigating Common-Class Bias](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Universal_Semi-Supervised_Domain_Adaptation_by_Mitigating_Common-Class_Bias_CVPR_2024_paper.pdf).

## Setup
Create a conda environment:
```bash
conda env create -f environment.yml
```

## Datasets

### Available Datasets
We prepared two public datasets:
- Office-Home
- DomainNet
```bash
python download_data.py
```

In the `data` directory, the `txt` folder contains the text files for the splits for each dataset, under the name of the dataset. `txt` folder is for covariate shift only, `txt_labelshift` folder is for covariate + label shift with same sample size as in `txt` folder, and `txt_fullsize` folder is for covariate + label shift with full dataset size. Generate splits by navigating to the selected folder and running
```bash
python generate_txt.py
```

### Adding New Dataset

#### Structure of data
To add new dataset (*e.g.,* NewData), it should be placed in a folder named `NewData` in the datasets directory (path provided in the arguments for main.py, `./data` by default).
The file structure for the dataset should be: 
```
NewData
│
└───domain1
│   │   image1
│   │   image2
│   │   ...
│   
└───domain2
│   │   image1
│   │   image2
│   │   ...    
│ 
...
```
#### Generating data splits
The splits for each domain is defined as **50% train, 20% validation, 30% test**.
Few-shot training and validation sets are sampled from the corresponding splits.

In the datasets directory, the `txt` folder contains the text files for the splits for each dataset, under the name of the dataset. `txt` folder is for covariate shift only, `txt_labelshift` folder is for covariate + label shift with same sample size as in `txt` folder, and `txt_fullsize` folder is for covariate + label shift with full dataset size.
Each row in the text file is in the format: `relative_path_of_image_to_dataset_folder class_id`.
(*e.g.,* `Clipart/Alarm_Clock/00053.jpg 0`). 

To generate the text files for NewData, after ensuring it has the file structure as stated above, create a new folder named `NewData` in the `txt` folder and run the provided `generate_txt.py`.

#### Configurations
Next, you have to add configs for the dataset in `configs/hparams.py`, `configs/data_model_configs.py` , `dataloader/dataloader.py` to define training hyperparameters and cross-domain adaptation scenarios.


## Domain Adaptation Algorithms

### Existing Algorithms
- Supervised baseline
- [CDAC](https://arxiv.org/abs/2104.09415)
- [PAC](https://www.bmvc2021-virtualconference.com/assets/papers/0764.pdf)
- [AdaMatch](https://arxiv.org/pdf/2106.04732.pdf)
- [DST](https://arxiv.org/abs/2202.07136)
- Proposed method

### Adding New Algorithm
To add a new algorithm, place it in `algorithms/algorithms.py`.


## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `--experiment_description`.
- Each experiment could have different trials, each is specified by `--run_description`.

### Training a model

To train a model:
```bash
python main.py  --experiment_description expt_run-txt-Resnet34-office_home-openpartial  \
                --run_description expt-Proposed-kshot-3 \
                --da_setting openpartial \
                --da_method Proposed \
                --dataset office_home \
                --backbone Resnet34 \
                --num_seeds 3 \
                --sampling kshot \
                --num_shots 3 \
                --data_path "./data/txt"
                --data_root "./data"
```

Sample scripts are in `scripts`.


### Deploying results on WandB Team
We use [Wandb](https://wandb.ai/) for visualizations of model training. 

Sign up for a WandB account using github or google account.
Add `--wandb_entity TEAM_NAME` as an argument to main.py where `TEAM_NAME` is an existing WandB team you are in. 
Eg. `--wandb_entity ssda`

Additional WandB arguments can be specified through `wandb_dir, wandb_project, wandb_tag` for organizing WandB runs, logs and artifacts.


## Results

Results for each run are saved in `experiments_logs`.
Obtain consolidated results by
```bash
python consolidation/consolidate_run.py
```

## Acknowledgement

This repository is adapted from [AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data](https://github.com/emadeldeen24/AdaTime).
