import os
import itertools
import numpy as np
from PIL import Image


def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class office_home():
    def __init__(self, wandb_tag):
        if ('debug' in wandb_tag):
            self.scenarios = [("Art", "Clipart")]
        elif ('first' in wandb_tag):
            self.scenarios = [("Art", "Clipart")]
        elif ('subset' in wandb_tag):
            self.scenarios = [("Art", "Clipart"), ("Clipart", "Product"), ("Product", "Real")]
        else:
            domains = ['Art', 'Clipart', 'Product', 'Real']
            self.scenarios = list(itertools.product(domains, domains))
            same_domain_scenarios = [(d, d) for d in domains]
            self.scenarios = [s for s in self.scenarios if s not in same_domain_scenarios]


class domain_net():
    def __init__(self, wandb_tag):
        if ('debug' in wandb_tag):
            self.scenarios = [("clipart", "painting")]
        if ('first' in wandb_tag):
            self.scenarios = [("clipart", "painting")]
        elif ('subset' in wandb_tag):
            self.scenarios = [("clipart", "painting"), ("painting", "real"), ("real", "sketch")]
        else:
            domains = ['clipart', 'painting', 'real', 'sketch']
            self.scenarios = list(itertools.product(domains, domains))
            same_domain_scenarios = [(d, d) for d in domains]
            self.scenarios = [s for s in self.scenarios if s not in same_domain_scenarios]
