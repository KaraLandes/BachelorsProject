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

class Train():
    def __init__(self, im_dir:str, network:nn.Module):
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
        
    def set_datasets(self, valid_share:float, test_share:float, dataset_type)->None:
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

        self.train_set = dataset_type(self.im_dir, train_ids, coefficient=1.5)
        self.test_set = dataset_type(self.im_dir, test_ids, seed=0)
        self.valid_set = dataset_type(self.im_dir, valid_ids, seed=1)

    def set_loaders(self, collate_fn_type, batch_size=1, workers=10)->None:
        """
        Setting dataloaders for trining
        :param batch_size: integer size of batch
        :param workers: integer number of workers
        :param collate_fn_type: collate fn function designed for different networks
        :return: None
        """
        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=batch_size,
                                       collate_fn=collate_fn_type,
                                       num_workers=workers,
                                       shuffle=True)
        self.valid_loader = DataLoader(dataset=self.valid_set,
                                       batch_size=batch_size,
                                       collate_fn=collate_fn_type,
                                       num_workers=workers,
                                       shuffle=False)
        self.test_loader = DataLoader(dataset=self.test_set,
                                      batch_size=batch_size,
                                      collate_fn=collate_fn_type,
                                      num_workers=workers,
                                      shuffle=False)

    def set_writer(self, log_dir)->None:
        """
        Setting tensorboard writer with specifig logs path
        :param log_dir: string path
        :return: None
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def set_device (self)->None:
        """
        Setting device for training with pytorch
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn_simple(self, batch):
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
        collated_msk = np.zeros(shape=(len(batch), target_shape[0], target_shape[1]))

        # fill with values
        for i, (im, msk) in enumerate(batch):
            collated_ims[i][:im.shape[0], :im.shape[1]] = im
            collated_msk[i][:msk.shape[0], :msk.shape[1]] = msk

        # normalise
        # TODO ???

        # tensors
        collated_msk = torch.from_numpy(collated_msk.astype(np.float32))
        collated_ims = torch.from_numpy(collated_ims.astype(np.float32))

        return collated_ims, collated_msk

    def collate_fn_rcnn(self, batch):
        """
        This collate fn function serves for Mask R-CNN model
        :param batch: databatch
        :return: TODO
        """

        #TODO SHAPES AND DATA STRUCTURES
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
            for m in [0,1]: #in every target dict there are 2 masks
                collated_msk = np.zeros(shape=(len(batch), desired_shape[0], desired_shape[1])) #suitible are
                collated_msk[:im.shape[0], :im.shape[1]] = target['masks'][m]#filling up with target infor
                target['masks'][m] = collated_msk
            target["masks"] = torch.from_numpy(target.masks)

        # tensors
        collated_ims = torch.from_numpy(collated_ims.astype(np.float32))
        collated_targets = tuple([target for _, target in batch])
        return collated_ims, collated_targets

    def run_epoch(self,loader: DataLoader, optimizer: torch.optim.Optimizer,
                  learningrate: float, weight_decay: float, optimize = True, loss=torch.nn.MSELoss()) -> list:
        """
        This is function which loops over loader once and does a forward pass trhough the network.
        @:return tuple of lists: images, masks, predictions, loss_values
        """
        loss_values = []
        i = 1
        for im, msk in tqdm(loader):
            im = im.to(self.device)
            msk = msk.to(self.device)
            optimizer.zero_grad()
            pred = self.net(im)

            l = loss(pred, msk)
            if optimize:
                try:
                    l.backward()
                    optimizer.step()
                except ValueError:
                    pass

            loss_values.append(l.detach().item())
            self.writer.add_scalars("BaseNet_Current_epoch_progress",
                                    {
                                        "Loss_of_sample": np.mean(loss_values),
                                    },
                                    global_step=i)
            self.writer.flush()
            i += 1
            torch.cuda.empty_cache()
        return loss_values

    def train(self, optimiser:torch.optim.Optimizer, save_model_path:str,
              save_images_path:str, epochs=10) -> None:
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
            train_results = self.run_epoch(loader=self.train_loader, optimizer=optimiser,
                                           learningrate=1e-3, weight_decay=1e-5)
            valid_results = self.run_epoch(loader=self.valid_loader, optimizer=optimiser,
                                           learningrate=1e-3, weight_decay=1e-5, optimize=False)

            mean_tr_loss, mean_val_loss = np.mean(train_results), np.mean(valid_results)
            train_losses.append(mean_tr_loss)
            valid_losses.append(mean_val_loss)

            torch.save(self.net.state_dict(),save_model_path+f"epoch{epoch}.pt")
            self.writer.add_scalars("BaseNet_Loss",
                               {"Training": mean_tr_loss,
                                "Validation": mean_val_loss},
                               global_step=epoch)
            print("Train Loss\t\t", "{:.2f}".format(mean_tr_loss))
            print("Validation Loss\t", "{:.2f}".format(mean_val_loss))
            self.eval.depict(loader=self.train_loader, network=self.net, name_convention=f"train_epoch{epoch}_", num=5,
                             writer=None, path=save_images_path, device=self.device)
            self.eval.depict(loader=self.valid_loader, network=self.net, name_convention=f"valid_epoch{epoch}_", num=5,
                             writer=self.writer, path=save_images_path, device=self.device)
            self.writer.flush()

        np.save(save_model_path+"train_losses.npy", train_losses)
        np.save(save_model_path+"valid_losses.npy", valid_losses)

class Evaluate():
    """A collection of different evaluation functions"""
    def __init__(self):
        pass

    def depict(self, loader, network, name_convention:str, path:str, device:str, writer:SummaryWriter, num=10) -> None:
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