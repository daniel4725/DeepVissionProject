import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import os
from utils import *
from train_and_eval import *
from models import *
from argparse import ArgumentParser
from datetime import datetime


if __name__ == '__main__':

    # set variables
    batch_size = 16
    num_workers = 0
    partial_data = 1  # part of the data to use - use all the data

    #  create the data loaders:
    # data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="M",
    #                                    partial_data=partial_data)
    # train_M_loader, valid_M_loader, test_M_loader = data_loaders
    #
    # data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="F",
    #                                    partial_data=partial_data)
    # train_F_loader, valid_F_loader, test_F_loader = data_loaders

    data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers,
                                       partial_data=1)
    train_loader, valid_loader, test_loader = data_loaders
    for ds, data_type in [(test_loader.dataset, "test"), (valid_loader.dataset, "validation"), (train_loader.dataset, "train")]:
        males = ds.metadata["Age"][ds.metadata["Gender"] == "M"]
        females = ds.metadata["Age"][ds.metadata["Gender"] == "F"]
        ax1 = males.hist(bins=30, alpha=0.5)
        ax2 = females.hist(bins=30, alpha=0.5)
        plt.legend({"males", "females"})
        plt.title(f"ages histogram - {data_type}")
        plt.xlabel("age")
        plt.ylabel("count")
        plt.show()
