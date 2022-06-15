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



# python3.6 evaluate_models.py
if __name__ == '__main__':
    # Select GPU
    device = set_GPU(0)
    model_type = "AgePredModel"  # AgePredModel_Gender_TotHyper_2to1_deep , AgePredModel_Gender_Hyper_2_5_1_relu
    experiment_name = "baseline_my_L2-0.05 F 2"
    cp_epoch_to_load = "52"
    only_graphs = False
    print(experiment_name, " epoch ", cp_epoch_to_load)

    # set variables
    batch_size = 16
    num_workers = 0
    partial_data = 1  # part of the data to use - use all the data

    # create the model
    use_gender_model = not ("AgePredModel" == model_type)
    model = locals()[model_type](device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015, betas=(0.9, 0.999))

    # checkpoints path and load the model
    cp_base_dir = "/media/rrtammyfs/Users/daniel/reaserch/brain age/checkpoints"
    cp_dir = os.path.join(cp_base_dir, experiment_name)
    checkpoint = load_checkpoint(os.path.join(cp_dir, f"{model.model_name}_e{str(cp_epoch_to_load)}"), model, optimizer)
    _, _, train_output, _, _ = checkpoint
    name = f"{experiment_name}- epoch {cp_epoch_to_load}"
    plotHist(train_output, model_name=name)
    if not only_graphs:
        #  create the data loaders:
        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="M", partial_data=partial_data)
        train_M_loader, valid_M_loader, test_M_loader = data_loaders

        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="F", partial_data=partial_data)
        train_F_loader, valid_F_loader, test_F_loader = data_loaders

        test_loader_FnM = [test_F_loader, test_M_loader]
        valid_loader_FnM = [valid_F_loader, valid_M_loader]
        train_loader_FnM = [train_F_loader, train_M_loader]

        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, partial_data=partial_data)
        train_loader, valid_loader, test_loader = data_loaders

        test_M_loss, test_M_mae = evaluate(model, test_M_loader, criterion, "test", device, use_gender_model=use_gender_model)
        print(f"\rtest Male-  loss: {test_M_loss:.3f},  MAE: {test_M_mae:.3f}")
        test_F_loss, test_F_mae = evaluate(model, test_F_loader, criterion, "test", device, use_gender_model=use_gender_model)
        print(f"\rtest Female-  loss: {test_F_loss:.3f},  MAE: {test_F_mae:.3f}")
        test_loss, test_mae = evaluate(model, test_loader, criterion, "test", device, use_gender_model=use_gender_model)
        print(f"\rtest mixed-  loss: {test_loss:.3f},  MAE: {test_mae:.3f}")

        valid_M_loss, valid_M_mae = evaluate(model, valid_M_loader, criterion, "valid", device, use_gender_model=use_gender_model)
        print(f"\rvalid Male-  loss: {valid_M_loss:.3f},  MAE: {valid_M_mae:.3f}")
        valid_F_loss, valid_F_mae = evaluate(model, valid_F_loader, criterion, "valid", device, use_gender_model=use_gender_model)
        print(f"\rvalid Female-  loss: {valid_F_loss:.3f},  MAE: {valid_F_mae:.3f}")
        valid_loss, valid_mae = evaluate(model, valid_loader, criterion, "valid", device, use_gender_model=use_gender_model)
        print(f"\rvalid mixed-  loss: {valid_loss:.3f},  MAE: {valid_mae:.3f}")

        train_loss_lst, train_mae_lst, valid_loss_lst, valid_mae_lst = train_output
        with open(os.path.join(cp_base_dir, 'results', name + '.txt'), 'w') as file:
            file.write(name + '\n')
            file.write(f"test:\n\tMale-  loss: {test_M_loss:.3f},  MAE: {test_M_mae:.3f}\n")
            file.write(f"\tFemale-  loss: {test_F_loss:.3f},  MAE: {test_F_mae:.3f}\n")
            file.write(f"\tmixed-  loss: {test_loss:.3f},  MAE: {test_mae:.3f}\n")
            file.write(f"validation:\n\tMale-  loss: {valid_M_loss:.3f},  MAE: {valid_M_mae:.3f}\n")
            file.write(f"\tFemale-  loss: {valid_F_loss:.3f},  MAE: {valid_F_mae:.3f}\n")
            file.write(f"\tmixed-  loss: {valid_loss:.3f},  MAE: {valid_mae:.3f}\n")

        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M")
        data = [dt_string, name,
                test_mae, test_M_mae, test_F_mae, valid_mae, valid_M_mae, valid_F_mae,
                test_loss, test_M_loss, test_F_loss, valid_loss, valid_M_loss, valid_F_loss]

        cols = ['datetime', 'experiment_name', 'test MAE - mixed', 'test MAE - M', 'test MAE - F',
                'val MAE - mixed', 'val MAE - M', 'val MAE - F',
                'test loss - mixed', 'test loss - M', 'test loss - F',
                'val loss - mixed', 'val loss - M', 'val loss - F']

        # save csv in the experiment's folder
        experiment_csv = pd.DataFrame(columns=cols)
        experiment_csv.loc[len(experiment_csv.index)] = data
        experiment_csv.to_csv(os.path.join(cp_dir, name + '.csv'), index=False)

        # add to the results csv
        results_csv_path = "/media/rrtammyfs/Users/daniel/reaserch/brain age/checkpoints/results/results.csv"
        results_csv = pd.read_csv(results_csv_path)
        results_csv.loc[len(results_csv.index), cols] = data
        results_csv.to_csv(results_csv_path, index=False)








