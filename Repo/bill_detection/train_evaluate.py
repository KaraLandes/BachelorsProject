from typing import List, Any

import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import time
from torch.utils.tensorboard import SummaryWriter
from .networks import BaseNet, MRCNN


class Train():
    def __init__(self, im_dir: str, network: nn.Module):
        self.im_dir = im_dir
        self.net = network

        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

        self.train_set = None
        self.test_set = None
        self.valid_set = None

        self.writer = None
        self.device = None

        self.eval = Evaluate()

    def set_datasets(self, valid_share: float, test_share: float, dataset_type, coefficient=1) -> None:
        """
        This method takes all images from self.imdir and create custom datasets
        :param dataset_type: Non instantiated class reference
        :param valid_share: percentage of data used for validation
        :param test_share: percentage of data used for testing
        :return:None
        """
        images = sorted(glob.glob(os.path.join(self.im_dir, "**", "*.jpg"), recursive=True))
        np.random.seed(0)
        indices = np.random.permutation(len(images))
        test_share, valid_share = int(test_share * len(images)), int(valid_share * len(images))
        train_share = len(images) - test_share - valid_share

        train_ids = indices[:train_share]
        test_ids = indices[train_share:(train_share + test_share)]
        valid_ids = indices[(train_share + test_share):]

        self.train_set = dataset_type(self.im_dir, train_ids, coefficient=coefficient)
        self.test_set = dataset_type(self.im_dir, test_ids, seed=0)
        self.valid_set = dataset_type(self.im_dir, valid_ids, seed=1)

    def set_loaders(self, batch_size=1, workers=10) -> None:
        """
        Setting dataloaders for trining
        :param batch_size: integer size of batch
        :param workers: integer number of workers
        :param collate_fn_type: collate fn function designed for different networks
        :return: None
        """
        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=batch_size,
                                       collate_fn=self.collate_fn,
                                       num_workers=workers,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_set,
                                       batch_size=batch_size,
                                       collate_fn=self.collate_fn,
                                       num_workers=workers,
                                       shuffle=False)
        self.test_loader = DataLoader(dataset=self.test_set,
                                      batch_size=batch_size,
                                      collate_fn=self.collate_fn,
                                      num_workers=workers,
                                      shuffle=False)

    def set_writer(self, log_dir) -> None:
        """
        Setting tensorboard writer with specifig logs path
        :param log_dir: string path
        :return: None
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def set_device(self) -> None:
        """
        Setting device for training with pytorch
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(self, batch):
        """
        Function to be redefined in every inherited class
        """
        raise ValueError("Please use inherited xlass to train a specific network")

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  optimize=True, loss=torch.nn.BCEWithLogitsLoss()) -> list:
        """
        Method to be redefined in every inherited class to train a specific Type of network
        """
        raise ValueError('please use an instance of inherited classes')

    def train(self, optimiser: torch.optim.Optimizer, save_model_path: str,
              save_images_path: str, epochs=10) -> None:
        """
        Running training loop with evaluation over every epoch.
        Updating tensorboard.
        Saving results
        :param epochs: Number of updates
        :return:
        """
        self.net.to(self.device)
        train_losses, valid_losses = [], []
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}:")
            time.sleep(1)
            train_results = self.run_epoch(loader=self.train_loader, optimizer=optimiser,)
            valid_results = self.run_epoch(loader=self.valid_loader, optimizer=optimiser, optimize=False)

            mean_tr_loss, mean_val_loss = np.mean(train_results), np.mean(valid_results)
            train_losses.append(mean_tr_loss)
            valid_losses.append(mean_val_loss)

            torch.save(self.net.state_dict(), save_model_path + f"epoch{epoch}.pt")
            self.writer.add_scalars("Network_Loss",
                                    {"Training": mean_tr_loss,
                                     "Validation": mean_val_loss},
                                    global_step=epoch)
            print("Train Loss\t\t", "{:.2f}".format(mean_tr_loss))
            print("Validation Loss\t", "{:.2f}".format(mean_val_loss))

            self.eval.evaluate(method=self.eval.depict, loader=self.train_loader, device=self.device,
                               network=self.net, path=save_images_path, naming=f"train_epoch_{epoch}")
            self.eval.evaluate(method=self.eval.depict, loader=self.valid_loader, device=self.device,
                               network=self.net, path=save_images_path, naming=f"valid_epoch_{epoch}")
            self.writer.flush()

        np.save(save_model_path + "train_losses.npy", train_losses)
        np.save(save_model_path + "valid_losses.npy", valid_losses)

class TrainBase(Train):
    def collate_fn(self, batch):
        """
        A collate function passed as argument to DataLoader.
        Brings batch to same dimensionality
        :return: data_batch, label_batch
        """
        max_r, max_c = 0, 0
        for im, msk in batch:
            if im.shape[0] > max_r: max_r = im.shape[0]
            if im.shape[1] > max_c: max_c = im.shape[1]
        target_shape = (max_r, max_c)

        # shape = elements in batch, rows, cols
        collated_ims = np.zeros(shape=(len(batch), target_shape[0], target_shape[1]))
        collated_msk = np.zeros(shape=(len(batch), 2, target_shape[0], target_shape[1]))

        # fill with values
        for i, (im, msk) in enumerate(batch):
            collated_ims[i][:im.shape[0], :im.shape[1]] = im
            for l in [0, 1]:
                collated_msk[i][l][:im.shape[0], :im.shape[1]] = msk[l]

        # normalise
        # TODO ???

        # tensors
        collated_msk = torch.from_numpy(collated_msk.astype(np.float32))
        collated_ims = torch.from_numpy(collated_ims.astype(np.float32))

        return collated_ims, collated_msk
    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  optimize=True, loss=torch.nn.BCEWithLogitsLoss()) -> list:
        """
        This is function which loops over loader once and does a forward pass through the network.
        @:return list of lost values of individual batches
        """
        loss_values = []
        i = 1
        for im, trg in tqdm(loader):
            optimizer.zero_grad()
            im = im.to(self.device)
            trg = trg.to(self.device)
            pred = self.net(im)

            l = loss(pred, trg)
            if optimize:
                try:
                    l.backward()
                    optimizer.step()
                except ValueError:
                    pass

            loss_values.append(l.detach().item())
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_sample": np.mean(loss_values),
                                    },
                                    global_step=i)
            self.writer.flush()
            i += 1
        for g in optimizer.param_groups:
            g['lr'] /= 2
        return loss_values

class TrainMask(Train):
    def collate_fn(self, batch):
        """
        This collate fn function serves for Mask R-CNN model
        :param batch: databatch
        :return: TODO
        """

        # TODO SHAPES AND DATA STRUCTURES
        max_r, max_c = 0, 0
        for im, _ in batch:
            if im.shape[0] > max_r: max_r = im.shape[0]
            if im.shape[1] > max_c: max_c = im.shape[1]
        desired_shape = (max_r, max_c)

        # shape = elements in batch, rows, cols
        collated_ims = np.zeros(shape=(len(batch), desired_shape[0], desired_shape[1]))

        # fill with values
        for i, (im, target) in enumerate(batch):
            collated_ims[i][:im.shape[0], :im.shape[1]] = im
            for m in [0, 1]:  # in every target dict there are 2 masks
                collated_msk = np.zeros(shape=(desired_shape[0], desired_shape[1]))  # suitable area
                collated_msk[:im.shape[0], :im.shape[1]] = np.copy(target['masks'][m])  # filling up with target infor
                target['masks'][m] = np.copy(collated_msk)
            target["masks"] = torch.from_numpy(np.copy(target["masks"]))

        # tensors
        collated_ims = [torch.from_numpy(image.reshape(1, max_r, max_c).astype(np.float32))
                        for image in collated_ims]
        collated_targets = tuple([target for _, target in batch])
        return collated_ims, collated_targets
    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  optimize=True, loss=torch.nn.BCEWithLogitsLoss()) -> list:
        """
        This is function which loops over loader once and does a forward pass trhough the network.
        @:return tuple of lists: images, masks, predictions, loss_values
        """
        loss_values = []
        i = 1
        for im, trg in tqdm(loader):
            optimizer.zero_grad()
            # moving to device
            im = [tensor.to(self.device) for tensor in im]
            for dictionary in trg:
                for key, tensor in dictionary.items():
                    dictionary[key] = tensor.to(self.device)
            if optimize:
                self.net.train()
                l = self.net(im, trg)['loss_mask']  # training model
            else:
                self.net.eval()
                pred = self.net(im)
                l = torch.tensor(0)


            loss_values.append(l.detach().item())
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_sample": np.mean(loss_values),
                                    },
                                    global_step=i)
            self.writer.flush()
            i += 1
            torch.cuda.empty_cache()

        return loss_values

class Evaluate():
    """A collection of different evaluation functions"""

    def __init__(self):
        pass

    def evaluate(self, method, loader: DataLoader, network: torch.nn.Module, device:str,
                 naming:str, path:str, num=2):
        allimages, allmasks, allpredictions = [], [], []
        for i, (images, targets) in enumerate(loader):
            if i == num: break
            if isinstance(network, torchvision.models.detection.MaskRCNN):
                # TODO !!!!!!!!
                network.eval()
                predictions = network(images.to(device))
                for j in range(len(images)):
                    image = images[j]
                    target = targets[j]
                    prediction = predictions[j]

                    scores = prediction['scores'].detach().numpy()
                    pr_masks = prediction['masks'].detach().numpy()
                    labels = prediction['labels'].detach().numpy()
                    # select best prediction for each label
                    for label in np.unique(labels):
                        score = np.where(labels == label, scores)

            elif isinstance(network, BaseNet):
                predictions = network(images.to(device))
                for j in range(len(images)):
                    image = images[j].to('cpu').detach().numpy()
                    target = targets[j].to('cpu').detach().numpy()

                    #softmaxing mask
                    prediction = predictions[j].to('cpu').detach().numpy()
                    max = np.max(prediction, axis=2, keepdims=True)  # returns max of each row and keeps same dims
                    e_x = np.exp(prediction - max)  # subtracts each row with its max value
                    sum = np.sum(e_x, axis=2, keepdims=True)  # returns sum of each row and keeps same dims
                    prediction = e_x / sum

                    allimages.append(image)
                    allmasks.append(target[0] + target[1])
                    allpredictions.append(prediction[0] + prediction[1])
            else:
                raise ValueError("For this network it's not clear how to post-process predictions")
        method(images=allimages, masks=allmasks, predictions=allpredictions,
               name_convention=naming, path=path)

    def depict(self, images, masks, predictions, name_convention: str, path: str) -> None:
        """
        This function saves plots and pictures of network performance
        """
        for i, image, mask, prediction in zip(range(len(images)), images, masks, predictions):
            true_masked = image * mask
            pred_masked = image * prediction
            plt.tight_layout()
            fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
            ax[0][0].imshow(image, cmap="binary_r")
            ax[0][0].set_title("True Image")
            ax[0][0].axis("off")
            ax[0][1].imshow(true_masked, cmap="binary_r")
            ax[0][1].set_title("Should be Masked")
            ax[0][1].axis("off")
            ax[0][2].imshow(pred_masked, cmap="binary_r")
            ax[0][2].set_title("Masked by model")
            ax[0][2].axis("off")
            temp = ax[1][1].imshow(mask*100, cmap="binary_r")
            ax[1][1].set_title("Actual Mask")
            ax[1][1].axis("off")
            fig.colorbar(temp, ax=ax[1][1], extend='both')
            temp = ax[1][2].imshow(prediction*100, cmap="binary_r")
            ax[1][2].set_title("Prediction Mask")
            ax[1][2].axis("off")
            fig.colorbar(temp, ax=ax[1][2], extend='both')
            ax[1][0].axis("off")
            fig.savefig(os.path.join(path, name_convention + "_imagenum_" +str(i).zfill(2) + ".png"), dpi=1200)
            plt.close('all')
            plt.cla()
            plt.clf()
