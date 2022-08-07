import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from ..train import Train


class TrainCorner(Train):
    def collate_fn(self, batch):
        """
        A collate function passed as argument to DataLoader.
        Brings batch to same dimensionality
        :return: data_batch, label_batch
        """
        collated_ims = []
        collated_corners = []
        collated_masks = []

        for i, b in enumerate(batch):
            im, corners, masks = b[0], b[1][0], b[1][1]
            im = [el / 255 for el in im]
            im = torch.tensor(im).to(torch.float32)
            collated_ims.append(im)
            collated_corners.append(torch.tensor(np.array(corners).flatten()).to(torch.float32))
            collated_masks.append(torch.tensor(masks).to(torch.float32))

        collated_ims = [el.expand(3, el.shape[-2], el.shape[-1]) for el in collated_ims]
        collated_ims = torch.stack(collated_ims)
        collated_corners = torch.stack(collated_corners)
        collated_masks = torch.stack(collated_masks)

        return collated_ims, collated_corners, collated_masks

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  save_model_path:str, epoch_num:int=None, optimize=True, criterion=torch.nn.MSELoss(reduction="mean"),
                  criterion2=torch.nn.NLLLoss()) -> list:
                  # this param. is additional for this network
        """
        This is function which loops over loader once and does a forward pass through the network.
        @:return list of lost values of individual batches
        """
        loss_values = []
        i = 1
        for im, corners_trg, mask_trg in tqdm(loader):

            if optimize: self.net.train()
            else: self.net.eval()

            im = im.to(self.device)
            corners_trg = (corners_trg+(torch.randn(len(corners_trg), 8)*2)).to(self.device)
            mask_trg = mask_trg.to(self.device).to(torch.long)
            coords_pred, mask_pred = self.net(im)

            l_corner = criterion(coords_pred, corners_trg)
            if self.net.compute_attention:
                l_mask = criterion2(mask_pred, mask_trg)
                l_total = l_corner+l_mask
            else:
                l_total = l_corner

            if optimize:
                optimizer.zero_grad()
                l_total.backward()
                optimizer.step()

            loss_values.append(l_corner.detach().item())
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_corners": l_corner.detach().item(),
                                        # "Loss_of_masks": l_mask.detach().item()
                                    },
                                    global_step=i)
            self.writer.flush()
            i += 1

        return loss_values

    def evaluate(self, method, loader: DataLoader, device: str,
                 naming: str, path: str, num=2):
        self.net.eval()
        allimages, alltargets, allpredictions = [], [], []
        count = 0
        for i, b in enumerate(loader):
            images, corners_trgs, masks_trgs = b
            images = images.to(self.device)
            corners_trgs = corners_trgs.to(self.device)
            masks_trgs = masks_trgs.to(self.device)
            predictions = self.net(images)

            for j in range(len(images)):
                if count == num: break
                image = images[j].to('cpu').detach().numpy()
                corners_trg = corners_trgs[j].to('cpu').detach().numpy()
                corners_pred = predictions[0][j].to('cpu').detach().numpy()

                mask_trg = masks_trgs[j].to('cpu').detach().numpy()
                if not self.net.compute_attention: masks_pred = np.zeros(mask_trg.shape)
                else:
                    masks_pred = predictions[1][j].to('cpu').detach().numpy()
                    masks_pred = masks_pred[1, :]

                allimages.append(image)
                alltargets.append((corners_trg, mask_trg))
                allpredictions.append((corners_pred, masks_pred))
                count += 1
                if count == num: break
            if count == num: break

        method(images=allimages, targets=alltargets, predictions=allpredictions,
               name_convention=naming, path=path)

    def depict_corners(self, images, targets, predictions, name_convention: str, path: str) -> None:
        """
        This function saves plots and pictures of network performance
        """
        for i, im, trg, pred in zip(range(len(images)), images, targets, predictions):
            im = im[0].reshape(im.shape[-2], im.shape[-1])
            corner_pred, mask_pred = pred
            corner_trg, mask_trg = trg

            fig, ax = plt.subplots(1, 3, figsize=(20*3, 20+7))
            ax[0].imshow(im, cmap="binary_r")

            colors = ['r','g','b','y']
            # bring to tuples
            corner_pred = [el if el > 0 else 0 for el in corner_pred]
            corner_pred = [el if el < 127 else 127 for el in corner_pred]
            corner_pred = [(corner_pred[0], corner_pred[1]), (corner_pred[2], corner_pred[3]),
                           (corner_pred[4], corner_pred[5]), (corner_pred[6], corner_pred[7])]
            corner_trg = [(corner_trg[0], corner_trg[1]), (corner_trg[2], corner_trg[3]),
                          (corner_trg[4], corner_trg[5]), (corner_trg[6], corner_trg[7])]

            for c, col in zip(corner_trg, colors): ax[0].scatter(c[0], c[1], c=col, s=1000, marker='o')
            for j, col in zip(range(4), colors):
                c = corner_pred[j]
                prev_c = corner_pred[j-1]
                ax[0].scatter(c[0], c[1], c=col, s=1000, marker='X')
                ax[0].plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")
            ax[0].set_title("True Image, Predictions - ☓, True - ◯", fontsize=40)
            pos = ax[0].get_position()
            ax[0].set_position([pos.x0, pos.y0 + pos.height * 0.1,
                                pos.width, pos.height * 0.9])
            ax[0].axis("off")

            ax[1].imshow(mask_pred, cmap="binary_r")
            ax[1].set_title("Masks prediction", fontsize=40)
            ax[1].axis("off")

            ax[2].imshow(mask_trg, cmap="binary_r")
            ax[2].set_title("Masks true", fontsize=40)
            ax[2].axis("off")


            plt.tight_layout(pad=2)
            fig.savefig(os.path.join(path, name_convention + "_imagenum_" + str(i).zfill(2) + ".png"), dpi=200)

            plt.close('all')
            plt.cla()
            plt.clf()
