from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os
from ..train import Train


class TrainBaseDetect(Train):
    def collate_fn(self, batch):
        """
        A collate function passed as argument to DataLoader.
        Brings batch to same dimensionality
        :return: data_batch, label_batch
        """
        max_r, max_c = batch[0][0].shape[0], batch[0][0].shape[1]
        target_shape = (max_r, max_c)

        # shape = elements in batch, rows, cols
        collated_ims = np.zeros(shape=(len(batch), 1, target_shape[0], target_shape[1]))
        collated_trg = np.zeros(shape=(len(batch), 4))

        # fill with values
        for i, (im, trg) in enumerate(batch):
            collated_ims[i][0][:im.shape[0], :im.shape[1]] = im
            collated_trg[i] = trg

        # tensors
        collated_trg = torch.from_numpy(collated_trg.astype(np.float32))
        collated_ims = torch.from_numpy(collated_ims.astype(np.float32))

        return collated_ims, collated_trg

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  save_model_path:str, epoch_num:int=None,  optimize=True,
                  criterion=torch.nn.MSELoss()) -> list:
        """
        This is function which loops over loader once and does a forward pass through the network.
        @:return list of lost values of individual batches
        """
        best_loss = np.Inf
        loss_values = []
        i = 1
        count_no_improvement = 0
        for im, trg in tqdm(loader):

            im = im.to(self.device)
            trg = trg.to(self.device)
            pred = self.net(im)

            l_box = criterion(pred, trg)
            if optimize:
                try:
                    optimizer.zero_grad()
                    l_box.backward()
                    optimizer.step()
                except ValueError:
                    pass
            trg *= im.shape[-1]
            pred *= im.shape[-1]
            l_box = criterion(pred, trg)

            loss_values.append(l_box.detach().item())
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_boxes": l_box.detach().item(),
                                    },
                                    global_step=i)
            self.writer.flush()

            #epoch early stopping
            if optimize and l_box.detach().item() < best_loss:
                count_no_improvement = 0
                best_loss = best_loss
            elif optimize:
                count_no_improvement += 1
            i += 1
        return loss_values

    def evaluate(self, method, loader: DataLoader, device: str,
                 naming: str, path: str, num=2):
        allimages, alltargets, allpredictions = [], [], []
        count = 0
        for i, (images, targets) in enumerate(loader):
            predictions = self.net(images.to(device))
            for j in range(len(images)):
                if count == num: break
                image = images[j].to('cpu').detach().numpy()
                target = targets[j].to('cpu').detach().numpy()
                target[0] *= image.shape[1]
                target[1] *= image.shape[1]
                target[2] *= image.shape[2]
                target[3] *= image.shape[2]
                prediction = predictions[j].to('cpu').detach().numpy()
                prediction[0] *= image.shape[1]
                prediction[1] *= image.shape[1]
                prediction[2] *= image.shape[2]
                prediction[3] *= image.shape[2]

                target = (target).astype(np.int64)
                prediction = (prediction).astype(np.int64)

                allimages.append(image)
                alltargets.append(target)
                allpredictions.append(prediction)
                count += 1
            if count == num: break

        method(images=allimages, targets=alltargets, predictions=allpredictions,
               name_convention=naming, path=path)
