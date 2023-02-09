import os

from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


class Train():
    def __init__(self, im_dir_gen: str, im_dir_real: str, im_dir_unseen: str, network: nn.Module):
        self.im_dir_gen = im_dir_gen
        self.im_dir_real = im_dir_real
        self.im_dir_unseen = im_dir_unseen
        self.net = network

        self.train_loader = None
        self.valid_loader = None

        self.train_set = None
        self.valid_set = None

        self.writer = None
        self.device = None

    def set_datasets(self, train_dataset_type, valid_dataset_type, coefficient=1, output_shape=(2000, 2000)) -> None:
        """
        This method takes all images from self.imdir and create custom datasets
        :param dataset_type: Non instantiated class reference
        :param valid_share: percentage of data used for validation
        :return:None
        """

        self.train_set = train_dataset_type(self.im_dir_gen, coefficient=coefficient, output_shape=output_shape)
        self.train_set_2 = valid_dataset_type(self.im_dir_real, output_shape=output_shape)
        self.valid_set = valid_dataset_type(self.im_dir_unseen, output_shape=output_shape, coefficient=3)

    def set_loaders(self, batch_size=1, workers=12) -> None:
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
        self.train_loader_2 = DataLoader(dataset=self.train_set_2,
                                       batch_size=2,
                                       collate_fn=self.collate_fn,
                                       num_workers=workers,
                                       shuffle=True)

        self.valid_loader = DataLoader(dataset=self.valid_set,
                                       batch_size=2,
                                       collate_fn=self.collate_fn,
                                       num_workers=workers,
                                       shuffle=False)

    def set_writer(self, log_dir) -> None:
        """
        Setting tensorboard writer with specific logs path
        :param log_dir: string path
        :return: None
        """
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=1)

    def set_device(self, cpu=False) -> None:
        """
        Setting device for training with pytorch
        """
        if cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(self, batch):
        """
        Function to be redefined in every inherited class
        """
        raise ValueError("Please use inherited class to train a specific network")

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  save_model_path:str, epoch_num:int, optimize=True,
                  criterion=torch.nn.MSELoss()) -> list:
        """
        Method to be redefined in every inherited class to train a specific Type of network
        """
        raise ValueError('Please use an instance of inherited classes')

    def train(self, optimizer: torch.optim.Optimizer, save_model_path: str,
              save_images_path: str, method, epochs=10) -> None:
        """
        Running training loop with evaluation over every epoch.
        Updating tensorboard.
        Saving results
        :param epochs: Number of updates
        :param optimizer type
        :param save_model_path Directory to save models pt
        :param save_images_path Directory to save images of training results
        :return:
        """
        self.net.to(self.device)
        train_losses, valid_losses = [], []
        best_loss = np.Inf
        count = 0

        loss_box = torch.nn.MSELoss()

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}: {'#'*100}")
            time.sleep(1)
            train_results = self.run_epoch(loader=self.train_loader, optimizer=optimizer,
                                           optimize=True,
                                           save_model_path=save_model_path, epoch_num=epoch,
                                           criterion=loss_box)
            torch.cuda.empty_cache()
            self.evaluate(method=method, loader=self.train_loader, device=self.device,
                          path=save_images_path, naming=f"train_epoch_{epoch}")
            # train_results = []


            train_results_2 = self.run_epoch(loader=self.train_loader_2, optimizer=optimizer,
                                             optimize=True,
                                             save_model_path=save_model_path, epoch_num=epoch,
                                             criterion=loss_box)
            torch.cuda.empty_cache()
            # self.evaluate(method=method, loader=self.train_loader_2, device=self.device,
            #               path=save_images_path, naming=f"train2_epoch_{epoch}")
            # train_results_2 = []
            valid_results = self.run_epoch(loader=self.valid_loader, optimizer=optimizer,
                                           optimize=False,
                                           save_model_path=save_model_path, epoch_num=None,
                                           criterion=loss_box)
            torch.cuda.empty_cache()
            self.evaluate(method=method, loader=self.valid_loader, device=self.device,
                          path=save_images_path, naming=f"valid_epoch_{epoch}")


            train_results = train_results+train_results_2
            mean_tr_loss, mean_val_loss = np.mean(train_results), np.mean(valid_results)
            train_losses.append(mean_tr_loss)
            valid_losses.append(mean_val_loss)

            # # decline lr after each epoch
            # current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            # space = np.linspace(5e-5, current_lr, 50)
            # for param_group in optimizer.param_groups: param_group['lr'] = space[-2]

            self.writer.add_scalars("Loss",
                                    {"Training": mean_tr_loss,
                                     "Validation": mean_val_loss},
                                    global_step=epoch)
            self.writer.flush()

            print("Train Loss\t", "{:.5f}".format(mean_tr_loss))
            print("Validation Loss\t", "{:.5f}".format(mean_val_loss))

            # early stopping
            if mean_val_loss < best_loss:
                torch.save(self.net.state_dict(), save_model_path + f"_on_ep{epoch}_new_best_model_{np.round(mean_val_loss,0)}.pt")
                best_loss = mean_val_loss
                count = 0
            else:
                count += 1
            if count >= int(epochs/3):
                print(f'Early stopping, no improvement on validation set during {count} epochs')
                break

        np.save(save_model_path + "train_losses.npy", train_losses)
        np.save(save_model_path + "valid_losses.npy", valid_losses)
        torch.save(self.net.state_dict(), save_model_path + f"_on_ep{epoch}_last_running_model.pt")

    def evaluate(self, method, loader: DataLoader, device: str,
                 naming: str, path: str, num=2):
        raise NotImplementedError("Write model specific function")

    def depict(self, images, targets, predictions, name_convention: str, path: str) -> None:
        """
        This function saves plots and pictures of network performance
        """
        for i, im, trg, pred in zip(range(len(images)), images, targets, predictions):
            # im = im.reshape(im.shape[-2], im.shape[-1])
            temp = np.zeros((im.shape[1], im.shape[1], 3))
            temp[:, :, 0] = im[0]
            temp[:, :, 1] = im[1]
            temp[:, :, 2] = im[2]
            temp = temp.astype(np.int)

            true_masked = temp[trg[0]:trg[1], trg[2]:trg[3]]  # cropping
            pred_masked = temp[pred[0]:pred[1], pred[2]:pred[3]]

            fig, ax = plt.subplots(1, 3, figsize=(24, 8))
            ax[0].imshow(temp)
            true_box = patches.Rectangle((trg[2], trg[0]), trg[3] - trg[2], trg[1] - trg[0],
                                         linewidth=1, edgecolor='g', facecolor='none',
                                         label=f"True box")
            pred_box = patches.Rectangle((pred[2], pred[0]), pred[3] - pred[2], pred[1] - pred[0],
                                         linewidth=1, edgecolor='r', facecolor='none',
                                         label=f"Predicted box")
            ax[0].add_patch(true_box)
            ax[0].add_patch(pred_box)
            ax[0].set_title("True Image", fontsize=30)
            pos = ax[0].get_position()
            ax[0].set_position([pos.x0, pos.y0 + pos.height * 0.1,
                                pos.width, pos.height * 0.9])
            ax[0].legend(loc='upper center', bbox_to_anchor=(.5, .05), fontsize=10)
            ax[0].axis("off")
            try:
                ax[1].imshow(true_masked)
            except:
                print(true_masked.shape)
                print()
            ax[1].set_title("Should be Masked", fontsize=30)
            ax[1].axis("off")

            try:
                ax[2].imshow(pred_masked)
            except:
                print(pred_masked.shape)
                print()
            ax[2].set_title("Masked by model", fontsize=30)
            ax[2].axis("off")
            plt.tight_layout(pad=2)
            fig.savefig(os.path.join(path, name_convention + "_imagenum_" + str(i).zfill(2) + ".png"), dpi=200)

            plt.close('all')
            plt.cla()
            plt.clf()
