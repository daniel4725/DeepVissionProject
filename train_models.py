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


def init_wandb(args):
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        config={
            "data_type": args.data_type,
            "partial_data": args.partial_data,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model": args.model,
            "lr": args.lr,
            "L2_lambda": args.L2,
            "dropout": args.dropout,
            "checkpoint_dir": args.checkpoint_dir,
            "continue_from_checkpoint": args.continue_from_checkpoint,
            "cp_epoch": args.cp_epoch,
            "transform": args.transform
        })

# python3.6 train_models.py --wandb_callback True --experiment_name "baseline_L2-0.1 M 1" --GPU 2 --model "AgePredModel" --data_type M --L2 0.1 --epochs 50
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--wandb_callback", choices=['True', 'False'], default="False")
    parser.add_argument("--experiment_name", default="0 tst")
    parser.add_argument("--GPU", default=0)

    # model : AgePredModel, AgePredModel_Gender_TotHyper_2to1_deep, AgePredModel_Gender_Hyper_nonLin_tanh_2_3_1_embedding
    parser.add_argument("--model", default="AgePredModel")
    parser.add_argument("--L2", default=0)
    parser.add_argument("--dropout", default=0.2)
    parser.add_argument("--lr", default=0.00015)

    # data
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--data_type", choices=['M', 'F', 'M&F', 'mixed'], default="F")
    parser.add_argument("--partial_data", default=1)
    parser.add_argument("--num_workers", default=1)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--transform", default=None)

    # checkpoints save dir
    parser.add_argument("--checkpoint_dir", default="/media/rrtammyfs/Users/daniel/reaserch/brain age/checkpoints")
    parser.add_argument("--save_cp_every", default=1)

    # resume learning from checkpoint option
    parser.add_argument("--continue_from_checkpoint", default=None)
    parser.add_argument("--cp_epoch", default=None)

    # project name dont change!
    parser.add_argument("--project_name", default="BrainAgeHyperNet")

    args = parser.parse_args()

    # ---------  wandb ----------
    wandb_callback = (args.wandb_callback == "True")
    if wandb_callback:
        # wandb.login(key='eb1e510a4bed996a9dac07bf3d3a2bda00cb113d')
        init_wandb(args)
    # --------------------------

    # Select GPU
    device = set_GPU(args.GPU)
    print("experiment name: ", args.experiment_name)
    print("epochs: ", args.epochs)
    print("data type: ", args.data_type)
    print("partial data: ", args.partial_data)

    # create the checkpoints dir
    experiment_name = args.experiment_name
    cp_base_dir = args.checkpoint_dir
    cp_dir = os.path.join(cp_base_dir, experiment_name)
    os.makedirs(cp_dir, exist_ok=True)

    # set variables
    transform = args.transform
    if transform is not None:
        transform = locals()[transform]

    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    partial_data = float(args.partial_data) # part of the data to use
    epochs = int(args.epochs)
    save_cp_every = int(args.save_cp_every)
    lr = float(args.lr)
    L2_lambda = float(args.L2)
    dropout = float(args.dropout)

    # create the model
    use_gender_model = not ("AgePredModel" == args.model)
    if not use_gender_model:
        model = locals()[args.model](device=device, dropout=dropout)
    else:
        model = locals()[args.model](device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=L2_lambda)

    if args.continue_from_checkpoint is None:
        checkpoint = None
    else:
        checkpoint = load_checkpoint(os.path.join(cp_dir, f"{model.name}_e{str(args.cp_epoch)}"), model, optimizer)

    #  create the relevant data loaders:
    if args.data_type in ["M", "M&F"]:
        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="M",
                                           partial_data=partial_data, transform=transform)
        train_M_loader, valid_M_loader, test_M_loader = data_loaders
        if args.data_type == "M":
            train_loader, valid_loader, test_loader = train_M_loader, valid_M_loader, test_M_loader

    if args.data_type in ["F", "M&F"]:
        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers, gender="F",
                                           partial_data=partial_data, transform=transform)
        train_F_loader, valid_F_loader, test_F_loader = data_loaders
        if args.data_type == "F":
            train_loader, valid_loader, test_loader = train_F_loader, valid_F_loader, test_F_loader
        else:
            test_loader = [test_F_loader, test_M_loader]
            valid_loader = [valid_F_loader, valid_M_loader]
            train_loader = [train_F_loader, train_M_loader]

    if args.data_type == "mixed":
        data_loaders = get_mri_dataloaders(batch_size=batch_size, num_workers=num_workers,
                                           partial_data=partial_data, transform=transform)
        train_loader, valid_loader, test_loader = data_loaders

    train_model(model, optimizer, criterion,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=epochs, device=device, continue_from_checkpoint=checkpoint,
                save_cp_every=save_cp_every, cp_dir=cp_dir, use_gender_model=use_gender_model,
                wandb_callback=wandb_callback, L2_lambda=False)

    if wandb_callback:
        wandb.finish()



