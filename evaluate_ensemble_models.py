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
    # Select GPU
    device = set_GPU(3)
    model_type = "AgePredModel"  # AgePredModel_Gender_TotHyper_2to1_deep , AgePredModel_Gender_Hyper_2_5_1_relu
    name = "baseline mixed ensemble"
    experiment_names = ["baseline 0.5mixed A_L2-1% 1", "baseline 0.5mixed A_L2-1% 2"]
    cp_epochs_to_load = ["56", "42"]
    print(experiment_names)
    print(cp_epochs_to_load)

    # set variables
    batch_size = 16
    num_workers = 0

    # create the model
    use_gender_model = not ("AgePredModel" == model_type)
    models_lst = []
    for _ in range(len(experiment_names)):
        tmp_model = locals()[model_type](device=device)
        models_lst.append(tmp_model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(models_lst[0].parameters(), lr=0.00015, betas=(0.9, 0.999))

    # checkpoints path and load the model
    model_name = models_lst[0].model_name
    cp_base_dir = "/media/rrtammyfs/Users/daniel/reaserch/brain age/checkpoints"
    for i in range(len(experiment_names)):
        cp_dir = os.path.join(cp_base_dir, experiment_names[i])
        checkpoint = load_checkpoint(os.path.join(cp_dir, f"{model_name}_e{str(cp_epochs_to_load[i])}"),
                                     models_lst[i], optimizer)
        _, _, train_output, _, _ = checkpoint

    model = EnsembleModel(models_lst, device=device)

    MAE_mixed_lst = []
    MAE_M_lst = []
    MAE_F_lst = []
    std_AbsErr_mixed_lst = []
    std_AbsErr_M_lst = []
    std_AbsErr_F_lst = []

    ages_cols = []
    jumps = 10
    for ages in zip(range(0, 91, jumps), range(jumps, 91, jumps)):
        print("\r", ages)
        ages_cols.append(str(ages[0]) + '-' + str(ages[1]))
        #  create the data loaders:
        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="M",
                                           partial_data=1, ages=ages)
        train_M_loader, valid_M_loader, test_M_loader = data_loaders

        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="F",
                                           partial_data=1, ages=ages)
        train_F_loader, valid_F_loader, test_F_loader = data_loaders

        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers,
                                           partial_data=1, ages=ages)
        train_loader, valid_loader, test_loader = data_loaders

        M_loader = [test_M_loader, valid_M_loader]
        F_loader = [test_F_loader, valid_F_loader]
        reg_loader = [test_loader, valid_loader]

        test_M_loss, test_M_mae, test_M_std_AE = evaluate(model, M_loader, criterion, "test", device,
                                                          use_gender_model=use_gender_model, return_std=True)
        test_F_loss, test_F_mae, test_F_std_AE = evaluate(model, F_loader, criterion, "test", device,
                                                          use_gender_model=use_gender_model, return_std=True)
        test_loss, test_mae, test_std_AE = evaluate(model, reg_loader, criterion, "test", device,
                                                    use_gender_model=use_gender_model, return_std=True)

        MAE_mixed_lst.append(test_mae)
        MAE_M_lst.append(test_M_mae)
        MAE_F_lst.append(test_F_mae)

        std_AbsErr_mixed_lst.append(test_std_AE)
        std_AbsErr_M_lst.append(test_M_std_AE)
        std_AbsErr_F_lst.append(test_F_std_AE)


    MAE_mixed_lst, std_AbsErr_mixed_lst = torch.Tensor(MAE_mixed_lst), torch.Tensor(std_AbsErr_mixed_lst)
    MAE_M_lst, std_AbsErr_M_lst = torch.Tensor(MAE_M_lst), torch.Tensor(std_AbsErr_M_lst)
    MAE_F_lst, std_AbsErr_F_lst = torch.Tensor(MAE_F_lst), torch.Tensor(std_AbsErr_F_lst)
    x_M = np.arange(len(MAE_M_lst))
    x_F = x_M + 0.1
    x_mixed = x_M + 0.2

    # plt.fill_between(ages_cols, MAE_F_lst - std_AbsErr_F_lst, MAE_F_lst + std_AbsErr_F_lst, color='tab:pink', alpha=0.2)
    # plt.plot(ages_cols, MAE_F_lst, c='tab:pink')
    plt.errorbar(x_F, MAE_F_lst, std_AbsErr_F_lst, c='tab:pink')
    plt.plot(x_F, MAE_F_lst, 'o', c='tab:pink')
    plt.plot(np.NaN, np.NaN, 'o', c='tab:pink', label='female')

    # plt.fill_between(ages_cols, MAE_M_lst - std_AbsErr_M_lst, MAE_M_lst + std_AbsErr_M_lst, color='b', alpha=0.2)
    # plt.plot(ages_cols, MAE_M_lst, c='b')
    plt.errorbar(x_M, MAE_M_lst, std_AbsErr_M_lst, c='b')
    plt.plot(x_M, MAE_M_lst, 'o', c='b')
    plt.plot(np.NaN, np.NaN, 'o', c='b', label='male')

    # plt.fill_between(ages_cols, MAE_mixed_lst - std_AbsErr_mixed_lst, MAE_mixed_lst + std_AbsErr_mixed_lst, color='g', alpha=0.2)
    # plt.plot(ages_cols, MAE_mixed_lst, c='g')
    plt.errorbar(x_mixed, MAE_mixed_lst, std_AbsErr_mixed_lst, c='g')
    plt.plot(x_mixed, MAE_mixed_lst, 'o', c='g')
    plt.plot(np.NaN, np.NaN, 'o', c='g', label='mixed')

    plt.legend(loc=2)
    plt.grid()
    plt.title("MAE-AGE: " + name)
    plt.xlabel("age")
    plt.ylabel("MAE")
    plt.xticks(x_F, ages_cols)

    plt.savefig("MAE-AGE_ " + name + '.png')
    plt.show()

