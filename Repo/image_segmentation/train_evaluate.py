from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
from multiprocessing import set_start_method

def run_epoch(loader: DataLoader,
              network: nn.Module,
              optimizer: torch.optim.Optimizer,
              learningrate: float,
              weight_decay: float,
              device: str,
              optimize = True,
              loss=torch.nn.MSELoss()) -> tuple:
    """
    This is function which loops over loader once and does a forward pass trhough the network.
    @:return tuple of lists: images, masks, predictions, loss_values
    """

    loss_values = []
    optimizer = optimizer(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    for im, msk in tqdm(loader):
        im = im.to(device)
        msk = msk.to(device)
        optimizer.zero_grad()
        pred = network(im)

        l = loss(pred, msk)
        if optimize:
            try:
                l.backward()
                optimizer.step()
            except ValueError:
                pass

        loss_values.append(l.detach().item())
        torch.cuda.empty_cache()

    return loss_values

def depict(loader, network, name_convention:str, path:str, num=10)-> None:
    """
    This function saves plots and pictures of network performance
    """
    i = 0
    for images, masks in loader:
        if i >= num:
            break
        i += 1
        predictions = network(images)
        for im, msk, pred in zip(images, masks, predictions):
            im = im.cpu().detach().numpy()
            msk = msk.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            true_masked = im * (msk/100)
            pred_masked = im * (pred/100)
            fig, ax = plt.subplots(1, 3)
            plt.axes("off")
            ax[0].imshow(im)
            ax[1].imshow(true_masked)
            ax[2].imshow(pred_masked)
            fig.save(os.path.join(path, name_convention + str(i).zfill(2) + "png"))
            plt.close('all')
            plt.cla()
            plt.clf()