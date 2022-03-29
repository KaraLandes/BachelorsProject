from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter


def run_epoch(loader: DataLoader,
              network: nn.Module,
              optimizer: torch.optim.Optimizer,
              learningrate: float,
              weight_decay: float,
              device: str,
              writer: SummaryWriter,
              optimize = True,
              loss=torch.nn.MSELoss()) -> tuple:
    """
    This is function which loops over loader once and does a forward pass trhough the network.
    @:return tuple of lists: images, masks, predictions, loss_values
    """

    loss_values = []
    optimizer = optimizer(network.parameters(), lr=learningrate, weight_decay=weight_decay)
    i = 1
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
        writer.add_scalars("BaseNet_Current_epoch_progress",
                           {
                               "Loss_of_sample": np.mean(loss_values),
                           },
                           global_step=i)
        writer.flush()
        i += 1
        torch.cuda.empty_cache()

    return loss_values

def depict(loader, network, name_convention:str, path:str, device:str, writer:SummaryWriter, num=10)-> None:
    """
    This function saves plots and pictures of network performance
    """
    i = 0
    for images, masks in loader:
        if i == num: break
        i += 1
        images = images.to(device)
        predictions = network(images)
        images = images.to('cpu')
        predictions = predictions.to('cpu')
        for im, msk, pred in zip(images, masks, predictions):
            im = im.cpu().detach().numpy()
            msk = msk.cpu().detach().numpy()

            pred = pred.cpu().detach().numpy()
            pred = pred.reshape((pred.shape[1], pred.shape[2]))/100
            pred = np.where(np.isclose(pred, 2), 2, pred)
            pred = np.where(np.isclose(pred, 1), 1, pred)
            pred = np.where(np.isclose(pred, 0), 0, pred)

            true_masked = im * (msk/100)
            pred_masked = im * (pred/100)
            plt.tight_layout()
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            ax[0].imshow(im, cmap="binary_r")
            ax[0].set_title("True Image")
            ax[0].axis("off")
            ax[1].imshow(true_masked, cmap="binary_r")
            ax[1].set_title("Should be Masked")
            ax[1].axis("off")
            ax[2].imshow(pred_masked, cmap="binary_r")
            ax[2].set_title("Masked by model")
            ax[2].axis("off")
            fig.savefig(os.path.join(path, name_convention + str(i).zfill(2) + ".png"))
            if not writer is None:
                writer.add_figure(name_convention + str(i).zfill(2), fig)
            plt.close('all')
            plt.cla()
            plt.clf()