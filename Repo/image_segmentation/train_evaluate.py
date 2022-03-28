from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os

def run_epoch(loader: DataLoader,
              network:nn.Module,
              optimizer: torch.optim.Optimizer,
              learningrate: float,
              weight_decay: float,
              optimize = True,
              loss = torch.nn.MSELoss()) -> tuple:
    """
    This is function which loops over loader once and does a forward pass trhough the network.
    @:return tuple of lists: images, masks, predictions, loss_values
    """
    loss_values = []
    predictions = []
    images = []
    masks = []
    optimizer = optimizer(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    for i, (im, msk) in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        pred = network(im)

        l = loss(pred, msk)
        if optimize:
            l.backward()
            optimizer.step()

        loss_values.append(l.item())
        predictions.append(pred)
        images.append(im)
        masks.append(msk)

    return images, masks, predictions, loss_values

def depict(images, masks, predictions, loss_values, name_convention:str, path:str, num=10)-> None:
    """
    This function saves plots and pictures of network performance
    """

    #form subsample of size 100
    indices = np.random.permutation(images.size()[0])[:num]
    images = np.array(images[indices])
    masks = np.array(masks[indices])
    predictions = np.array(predictions[indices])
    loss_values = np.array(loss_values[indices])

    #plot
    for im, msk, pred, l, i in zip(images, masks, predictions, loss_values, range(num)):
        true_masked = im * (msk/100)
        pred_masked = im * (pred/100)
        fig, ax = plt.subplots(1,3)
        plt.axes("off")
        ax[0].imshow(im)
        ax[1].imshow(true_masked)
        ax[2].imshow(pred_masked)
        fig.save(os.path.join(path, name_convention+str(i).zfill(2)+"png"))
        plt.close('all')
        plt.cla()
        plt.clf()