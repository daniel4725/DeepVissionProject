import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import os
import wandb
from utils import *


def train_model(model, optimizer, criterion, train_loader, valid_loader, epochs, device,
                continue_from_checkpoint=None, save_cp_every=1000, cp_dir=None,
                wandb_callback=None):

    if cp_dir is None:
        raise ValueError("please input cp_dir to train_model function !!")

    if continue_from_checkpoint is None:
        train_loss_lst = []
        valid_loss_lst = []
        valid_mae_lst = []
        train_mae_lst = []
        start_epoch = 1
    else:
        _, epoch, loss, _, _ = continue_from_checkpoint
        start_epoch = epoch + 1
        train_loss_lst, train_mae_lst, valid_loss_lst, valid_mae_lst = loss

    show_timer = time.time()  # timer for progress printing
    train_start = time.time()  # timing the training
    train_size = len(train_loader)
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_time = time.time()  # timing each epoch
        model.train()

        # ------------ training data --------------------
        epoch_loss = 0
        num_samples = 0
        # epoch_abs_err_lst = torch.Tensor([]).to(device)

        for batch_idx, (x_batch, y_batch) in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).type(torch.float32)

            # update parameters and calculate the loss
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            num_samples += len(output)
            epoch_loss += loss.item() * len(output)
            iou = IOU(output, y_batch)

            # epoch_abs_err_lst = torch.cat([epoch_abs_err_lst, abs(output - y_batch)])

            # ---------- wandb --------------
            if wandb_callback:
                metrics = {"train/train_loss": loss.item(),
                           "train/epoch": epoch - 1
                           # "train/train_MAE": batch_mae # TODO
                           }
                wandb.log(metrics)
            # --------------------------------

            if time.time() - show_timer > 1 or (batch_idx == train_size - 1):  # enters every 1 sec,last batch
                show_timer = time.time()
                print(f"\rEpoch {epoch}:  [{batch_idx + 1}/{train_size}]"
                      f'train loss: {epoch_loss / num_samples:.3f}'
                      f'\ttrain MAE: {1:.3f}', end='')  # TODO

        # print in the end of the training epoch and append the losses
        print(f'\rEpoch {epoch}:\ttrain loss: {epoch_loss / num_samples:.3f}'
              f'\ttrain MAE: {1:.3f}')  # TODO
        train_loss_lst.append(epoch_loss / num_samples)
        # train_mae_lst.append(epoch_abs_err_lst.mean())  TODO

        # ------------ validation data --------------------
        model.eval()  # evaluating the validation data
        valid_loss, valid_measures = evaluate(model, valid_loader, criterion, "validation", device)
        valid_loss_lst.append(valid_loss)
        # valid_mae_lst.append(valid_mae)  # TODO valid_measures

        # ---------- wandb --------------
        if wandb_callback:
            metrics = {"train/train_loss": loss.item(),
                       "train/epoch": epoch,
                       # "train/train_MAE": batch_mae   # TODO
                       }
            wandb.log(metrics)
            val_metrics = {"val/val_loss(MSE)": valid_loss,
                           # "val/val_MAE": valid_mae  # TODO
                           }
            wandb.log({**metrics, **val_metrics})
        # --------------------------------


        print(f'\r\t\t\tvalid loss: {valid_loss:.3f}  valid MAE: {1:.3f}'  # TODO valid mae
              f'\tepoch time: {show_time(time.time() - epoch_time)}\n')

        if (epoch % save_cp_every) == 0:
            path = os.path.join(cp_dir, model.model_name + "_e" + str(epoch))
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch,
                loss=(train_loss_lst, train_mae_lst, valid_loss_lst, valid_mae_lst),  # TODO
            )

    print(f'Finished Training: {show_time(time.time() - train_start)}')
    return train_loss_lst, train_mae_lst, valid_loss_lst, valid_mae_lst  # TODO what output?


def evaluate(model, data_loader, criterion, data_type, device):
    model.eval()
    epoch_loss = 0
    num_samples = 0

    data_size = len(data_loader)

    show_timer = time.time()  # timer for progress printing
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).type(torch.float32)

            output = model(x_batch)
            loss = criterion(output, y_batch)
            num_samples += len(output)
            epoch_loss += loss.item() * len(output)

            # abs_err_lst = torch.cat([abs_err_lst, abs(output - y_batch)])  # TODO

            if time.time() - show_timer > 1 or (batch_idx == data_size - 1):  # enters every 1 sec,last batch
                show_timer = time.time()
                print('\r{}:[{}/{}]\t  loss: {:.3f}  MAE: {:.3f}'.format(
                data_type, (batch_idx + 1), data_size,
                    epoch_loss/num_samples, 11111111), end='')  # TODO

    return epoch_loss/num_samples, abs_err_lst.mean()

if __name__ == '__main__':
    from models import *
    from MRI_Dataset import MRIDataset
    from torch.utils.data import Dataset, DataLoader

    # Select GPU
    GPU_ID = '3'
    print('GPU USED: ' + GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if runes on one

    cp_dir = "/media/rrtammyfs/Users/daniel/reaserch/checkpoints/hyper_with_embedding2to1"
    os.makedirs(cp_dir, exist_ok=True)

    batch_size = 16
    num_workers = 4

    valid_M_ds = MRIDataset(gender="M", data_type="valid")
    test_M_ds = MRIDataset(gender="M", data_type="test")
    train_M_ds = MRIDataset(gender="M", data_type="train")
    valid_F_ds = MRIDataset(gender="F", data_type="valid")
    test_F_ds = MRIDataset(gender="F", data_type="test")
    train_F_ds = MRIDataset(gender="F", data_type="train")

    train_M_loader = DataLoader(dataset=train_M_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_M_loader = DataLoader(dataset=valid_M_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_M_loader = DataLoader(dataset=test_M_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_F_loader = DataLoader(dataset=train_F_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_F_loader = DataLoader(dataset=valid_F_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_F_loader = DataLoader(dataset=test_F_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = AgePredModel_Gender_TotHyper_2to1(device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015, betas=(0.9, 0.999))

    # descrip, epoch, loss, acc, other = load_checkpoint(os.path.join(cp_dir, "age_prediction_e2"), model, optimizer)

    train_model(model, optimizer, criterion, (train_M_loader, train_F_loader),
                valid_loader=(valid_M_loader, valid_F_loader), EPOCHS=20,
                device=device, save_cp_every=1, cp_dir=cp_dir)
