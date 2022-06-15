import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import os
from MRI_Dataset import *

def set_GPU(gpu_num):
    GPU_ID = str(gpu_num)
    print('GPU USED: ' + GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if runes on one
    return device

def plotHist(training_output, model_name="", slice=(None, None)):
    """ plots the learning curves w.r.t the epochs """

    train_loss_lst, train_mae_lst, valid_loss_lst, valid_mae_lst = training_output

    x = [i+1 for i in range(len(train_loss_lst))]

    # plot loss
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch number', ylabel='MSE',
           title=model_name+': loss over epochs')
    ax.plot(x[slice[0]:slice[1]], train_loss_lst[slice[0]:slice[1]], c='b')
    ax.plot(np.NaN, np.NaN, c='b', label='train')
    ax.plot(x[slice[0]:slice[1]], valid_loss_lst[slice[0]:slice[1]], c='r')
    ax.plot(np.NaN, np.NaN, c='r', label='validation')
    ax.legend(loc=1)
    ax.grid()

    # plot mean average error
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch number', ylabel='MAE',
           title=model_name+': MAE over epochs')
    ax.plot(x[slice[0]:slice[1]], train_mae_lst[slice[0]:slice[1]], c='b')
    ax.plot(np.NaN, np.NaN, c='b', label='train')
    ax.plot(x[slice[0]:slice[1]], valid_mae_lst[slice[0]:slice[1]], c='r')
    ax.plot(np.NaN, np.NaN, c='r', label='validation')
    ax.legend(loc=1)
    ax.grid()

    plt.show()


def save_checkpoint(path, model, optimizer, epoch, description="No description", loss=(), accuracy=(), other=()):
    """ saves the current state of the model and optimizer and the training progress"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'description': description,
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'other': other
    }, path)


def load_checkpoint(path, model, optimizer):
    """ loads the state of the model and optimizer and the training progress"""
    cp = torch.load(path)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    return cp['description'], cp['epoch'], cp['loss'], cp['accuracy'], cp['other']


def show_time(seconds):
    time = int(seconds)
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    if day != 0:
        return "%dD %dH %dM %dS" % (day, hour, minutes, seconds)
    elif day == 0 and hour != 0:
        return "%dH %dM %dS" % (hour, minutes, seconds)
    elif day == 0 and hour == 0 and minutes != 0:
        return "%dM %dS" % (minutes, seconds)
    else:
        return "%dS" % (seconds)


def normalize_minmax(img):
    img = img - img.min()
    img = img / img.max()
    return img


def normalize_stdmean(img):
    img = img - img.mean()
    img = img / img.std()
    return img


def IOU(output, seg):
    return 1

if __name__ == '__main__':
    data_loaders = get_mri_dataloaders(batch_size=16, num_workers=0, gender="F", data_in_storage_server=False,
                                       transform=None)
    train_loader, valid_loader, test_loader = data_loaders
    img, _, _ = test_loader.dataset.__getitem__(1)
    print(img.max())
    print(img.min())
    print(img.mean())
    print(img.std())