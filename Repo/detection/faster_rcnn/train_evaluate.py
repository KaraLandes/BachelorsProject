from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from torchvision.ops import box_iou
from ..train import Train


class TrainFRCNN(Train):
    def collate_fn(self, batch):
        """
        A collate function passed as argument to DataLoader.
        Brings batch to same dimensionality
        :return: data_batch, label_batch
        """
        collated_ims = []
        collated_trg_dicts = []

        for i, (im, trg) in enumerate(batch):
            # im = np.reshape(im, (1, im.shape[0], im.shape[1]))
            collated_ims.append(torch.tensor(im).to(torch.float32))
            collated_trg_dicts.append(trg)

        return collated_ims, collated_trg_dicts

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, save_model_path: str,
                  epoch_num: int = None, optimize=True, criterion=torch.nn.MSELoss()) -> list:
        """
        This is function which loops over loader once and does a forward pass through the network.
        @:return list of lost values of individual batches
        """

        best_loss = np.Inf
        loss_values = []
        i = 1
        count_no_improvement = 0
        for im, trg in tqdm(loader):

            im = [el.to(self.device) for el in im]
            trg = [{k: v.to(self.device) for k, v in t.items()} for t in trg]
            pred, l_box = self.net((im, trg, optimize))

            # I take the best prediction
            # sometimes no element recognised, and there are 0 boxes predicted
            pred_boxes = []
            for pred_el, trg_el in zip(pred, trg):
                if len(pred_el['boxes']) > 0:
                    best_box_id = 0
                    pred_boxes.append(pred_el['boxes'][best_box_id])
                else:
                    pred_boxes.append(torch.tensor([0, 0, 0, 0]).to(torch.float32).to(self.device))

            pred_boxes = torch.stack(pred_boxes)
            trg_boxes = torch.flatten(torch.stack([el['boxes'] for el in trg]), 1)
            mse_l_box = criterion(pred_boxes, trg_boxes)
            l_box = sum(loss for loss in l_box.values())

            if optimize:
                try:
                    optimizer.zero_grad()
                    l_box.backward()
                    optimizer.step()
                except ValueError:
                    pass

            loss_values.append(mse_l_box.detach().item())
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_boxes": mse_l_box.detach().item(),
                                    },
                                    global_step=i)
            self.writer.flush()

            # epoch early stopping
            if optimize and l_box.detach().item() < best_loss:
                count_no_improvement = 0
                best_loss = best_loss
            elif optimize:
                count_no_improvement += 1
            if count_no_improvement > 100:
                print("No improvement for 1000 batches, epoch terminated.")
                break
            i += 1

        return loss_values

    def evaluate(self, method, loader: DataLoader, device: str,
                 naming: str, path: str, num=2):
        allimages, alltargets, allpredictions = [], [], []
        count = 0
        for i, (images, targets) in enumerate(loader):
            images = [el.to(self.device) for el in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            predictions, _ = self.net((images, targets, False))
            for j in range(len(images)):
                if count == num: break
                image = images[j].to('cpu').detach().numpy()

                target = targets[j]['boxes'].to('cpu').detach().numpy()[0].astype(int)
                target[1], target[2] = target[2], target[1]  # bring to x0,x1,y0,y1 as i have in depict

                prediction = predictions[j]['boxes'].to('cpu').detach().numpy().astype(int)

                # ious = []
                # if len(prediction) > 0:
                #     for box in prediction:
                #         iou = box_iou(torch.unsqueeze(torch.tensor(box), 0), torch.unsqueeze(torch.tensor(target), 0)).flatten()[0]
                #         ious.append(iou)
                #     best_box_id = torch.argmax(torch.stack(ious))
                #     prediction = prediction[best_box_id]
                # else:
                #     prediction = np.array([0, 0, 100, 100])
                if len(prediction) > 0:
                    best_box_id = 0
                    prediction = prediction[best_box_id]
                else:
                    prediction = np.array([0, 0, 100, 100])

                prediction[1], prediction[2] = prediction[2], prediction[1]  # bring to x0,x1,y0,y1 as i have in depict

                allimages.append(image)
                alltargets.append(target)
                allpredictions.append(prediction)
                count += 1
            if count == num: break

        method(images=allimages, targets=alltargets, predictions=allpredictions,
               name_convention=naming, path=path)
