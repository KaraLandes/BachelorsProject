import operator
import os

import cv2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from ..train import Train
from ..corners_nn.network import CornerDetector


class TrainRefine(Train):
    def collate_fn(self, batch):
        """
        A collate function passed as argument to DataLoader.
        Brings batch to same dimensionality
        :return: data_batch, label_batch
        """
        collated_ims = []
        collated_corners = []
        collated_masks = []
        collated_originals = []
        collated_corners_big = []

        for i, b in enumerate(batch):
            im, corners, masks, original, corners_big = b[0], b[1][0], b[1][1], b[1][2], b[1][3]

            im = [el / 255 for el in im]
            im = torch.tensor(im).to(torch.float32)
            collated_ims.append(im)

            original = [el / 255 for el in original]
            original = torch.tensor(original).to(torch.float32)
            collated_originals.append(original)

            collated_corners.append(torch.tensor(np.array(corners).flatten()).to(torch.float32))
            collated_corners_big.append(torch.tensor(np.array(corners_big).flatten()).to(torch.float32))
            collated_masks.append(torch.tensor(masks).to(torch.float32))

        collated_ims = [el.expand(3, el.shape[-2], el.shape[-1]) for el in collated_ims]
        collated_ims = torch.stack(collated_ims)

        collated_originals = [el.expand(3, el.shape[-2], el.shape[-1]) for el in collated_originals]
        collated_originals = torch.stack(collated_originals)
        collated_corners = torch.stack(collated_corners)
        collated_corners_big = torch.stack(collated_corners_big)
        collated_masks = torch.stack(collated_masks)

        return collated_ims, collated_corners, collated_masks, collated_originals, collated_corners_big

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                  save_model_path: str, epoch_num: int = None, optimize=True,
                  criterion=torch.nn.MSELoss(reduction="mean"),
                  criterion2=torch.nn.NLLLoss()) -> list:
        # this param. is additional for this network
        """
        This is function which loops over loader once and does a forward pass through the network.
        @:return list of lost values of individual batches
        """
        joint_corner_net = self.set_JCD()
        loss_values = []
        i = 1
        for im, _, _, originals, corners_big in tqdm(loader):
            # first we predict corners for images with previously trained net
            im = im.to(self.device)
            coords_pred, _ = joint_corner_net(im)

            # recalculation of predicted corners
            coords_pred = torch.reshape(coords_pred, shape=(len(coords_pred), 4, 2))
            corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))

            big_corners_pred = torch.tensor([[self.rescale_corners(c, im, originals)
                                              for c in el] for el in coords_pred])

            # refinement loop
            loss_values_per_sample = []
            for j in range(len(im)):
                try:
                    # perform initial crop
                    patches, old_prediction, target_refine_net, \
                    target_general, offsets = self.initial_crop(originals=torch.stack([originals[j]]),
                                                                big_corners_pred=torch.stack([big_corners_pred[j]]),
                                                                corners_big=torch.stack([corners_big[j]]),
                                                                output_shape=(50, 50),
                                                                window=200)
                    p = patches[0].to(self.device)
                    o_p = old_prediction[0]
                    n_p = self.net(p)
                    t_r_n = target_refine_net[0]
                    changes = {}

                    changes[f"{0:02d} offset"] = torch.tensor(offsets[0][0])
                    changes[f"{1:02d} rescale"] = {"to": offsets[0][1][1], "from": offsets[0][1][0]}
                    changes[f"{2:02d} prediction"] = torch.clone(n_p.detach())
                    if optimize:
                        l_corner = criterion(n_p.float(), t_r_n.float()).float()
                        optimizer.zero_grad()
                        l_corner.backward()
                        optimizer.step()
                    else:
                        n_p, changes = self.refine_loop(changes, n_p, o_p, t_r_n, p)

                    # when no more cropping happens I retrack the most current prediction to the original size and calculate loss
                    sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
                    pred_point = torch.clone(n_p.detach())
                    pred_point = self.untracking_changes(sorted_changes, pred_point)
                    loss_values_per_sample.append(criterion(pred_point, target_general[0]).item())
                except:
                    print("exception", "batch", i, ', image', j)
                    continue

            loss_values.append(np.mean(loss_values_per_sample))
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_corners": np.mean(loss_values[-1]),
                                    },
                                    global_step=i)
            self.writer.flush()
            i += 1

        return loss_values

    def untracking_changes(self, sorted_changes, point):
        for k, v in sorted_changes:
            if "offset" in k:
                point = point + v.to(self.device)
            elif "rescale" in k:
                point = torch.stack(self.rescale_corners(point,
                                                         current_image=v["to"],
                                                         new_image=v["from"]))  # reverse transformation
        return point

    def set_JCD(self):
        state_dict = torch.load(os.path.join("progress_tracking", "detection", "corners_nn",
                                             "models", "run_80_1",
                                             "corners_nn_on_ep17_new_best_model_32.0.pt"),
                                map_location=torch.device('cpu'))
        joint_corner_net = CornerDetector(compute_attention=True)
        joint_corner_net.load_state_dict(state_dict)
        joint_corner_net = joint_corner_net.to(self.device)
        return joint_corner_net

    def rescale_corners(self, corners, current_image, new_image):
        # dims [y, x]
        current_dims = current_image.shape[-2:]
        new_dims = new_image.shape[-2:]
        relative_corners = [corners[0] / current_dims[1], corners[1] / current_dims[0]]
        scaled_corners = [relative_corners[0] * new_dims[1], relative_corners[1] * new_dims[0]]
        return scaled_corners

    def refine_loop(self, changes, n_p, o_p, t_r_n, p):
        count = 3
        condition1 = sum((n_p - o_p) ** 2) > 1
        # condition2 = (count / 3) <= 10 if optimize else (count / 3) <= 300
        condition2 = (count / 3) <= 15
        while condition1 and condition2:
            # if optimize: self.net.train()
            # else: self.net.eval()
            # l_corner = criterion(n_p.float(), t_r_n.float()).float()
            # if optimize:
            #     optimizer.zero_grad()
            #     l_corner.backward()
            #     optimizer.step()

            for i in range(len(n_p)):
                if n_p[i] < 0: n_p[i] = torch.tensor(1)
                if n_p[i] > (p.shape[-1] - 1): n_p[i] = torch.tensor(p.shape[-1] - 1)

            delta_y = n_p[1] - o_p[1]
            delta_x = n_p[0] - o_p[0]
            if delta_x >= 0:
                x_start = abs(delta_x).item()/1
                x_end = p.shape[-1]
            else:
                x_start = 0
                x_end = (p.shape[-1] - abs(delta_x)/1).item()
            if delta_y >= 0:
                y_start = abs(delta_y).item()/1
                y_end = p.shape[-2]
            else:
                y_start = 0
                y_end = (p.shape[-2] - abs(delta_y)/1).item()

            patch_cropped = p[:, int(y_start):int(y_end), int(x_start):int(x_end)].detach().to("cpu").numpy()[0]

            t_r_n = t_r_n - torch.tensor([x_start, y_start]).to(self.device)
            changes[f"{count:02d} offset"] = torch.tensor([x_start, y_start])
            n_p = n_p - torch.tensor([x_start, y_start]).to(self.device)
            patch_cropped_resized = cv2.resize(patch_cropped[0], dsize=(50, 50), interpolation=cv2.INTER_AREA)
            t_r_n = torch.stack(self.rescale_corners(t_r_n,
                                                     current_image=patch_cropped,
                                                     new_image=patch_cropped_resized))
            changes[f"{(count + 1):02d} rescale"] = {"to": patch_cropped_resized, "from": patch_cropped}
            n_p = torch.stack(self.rescale_corners(n_p,
                                                   current_image=patch_cropped,
                                                   new_image=patch_cropped_resized))
            patch_cropped_resized = torch.tensor([patch_cropped_resized]).to(self.device)

            o_p = torch.clone(n_p.detach())

            n_p = self.net(patch_cropped_resized)

            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(patch_cropped_resized[0].detach().to("cpu").numpy(), cmap="binary_r")
            # ax.scatter(t_r_n[0].detach().to("cpu").numpy(), t_r_n[1].detach().to("cpu").numpy(), marker="^", c="green")
            # ax.scatter(n_p[0].detach().to("cpu").numpy(), n_p[1].detach().to("cpu").numpy(), marker="o", c="red")
            # ax.scatter(o_p[0].detach().to("cpu").numpy(), o_p[1].detach().to("cpu").numpy(), marker="o", c="yellow")
            # fig.savefig(f"chain_temp{count:02d}.png")
            # plt.close()

            p = torch.clone(patch_cropped_resized.detach())
            changes[f"{(count + 2):02d} prediction"] = torch.clone(n_p.detach())
            count += 3

            condition1 = sum((n_p - o_p) ** 2) > 1
            # condition2 = (count / 3) <= 10 if optimize else (count / 3) <= 300
            condition2 = (count / 3) <= 15

        return n_p, changes

    def initial_crop(self, originals: torch.Tensor, big_corners_pred: torch.Tensor,
                     corners_big: torch.Tensor, output_shape=(50, 50), window=70):

        colors = np.array(['red', 'green', 'blue', 'yellow'])
        active_corner_id = np.argwhere(colors == self.net.net_type)[0][0]

        patches = []
        current_prediction = []
        target_refine_net = []
        target_general = []
        offsets = []

        temp_count = 0
        for el, c_p_on_image, c_t_on_image in zip(originals, big_corners_pred, corners_big):
            c_p_on_image = c_p_on_image[active_corner_id]
            c_t_on_image = c_t_on_image[active_corner_id]

            # form the patch
            y_start = (c_p_on_image[1] - window).to(torch.int) if c_p_on_image[1] - window >= 0 else 0
            y_end = (c_p_on_image[1] + window).to(torch.int) if c_p_on_image[1] + window < originals.shape[-1] else -1
            x_start = (c_p_on_image[0] - window).to(torch.int) if c_p_on_image[0] - window >= 0 else 0
            x_end = (c_p_on_image[0] + window).to(torch.int) if c_p_on_image[0] + window < originals.shape[-1] else -1
            p = el[0,
                   y_start:y_end,
                   x_start:x_end]
            # rescale patch
            p_small = cv2.resize(p.detach().to("cpu").numpy(), dsize=output_shape, interpolation=cv2.INTER_AREA)
            # define coordinates of target and predicted corner on the patch
            c_p_on_patch = c_p_on_image - torch.tensor([x_start, y_start])
            c_t_on_patch = c_t_on_image - torch.tensor([x_start, y_start])

            # rescale corners
            c_p_on_patch_small = self.rescale_corners(c_p_on_patch, p, p_small)
            c_t_on_patch_small = self.rescale_corners(c_t_on_patch, p, p_small)

            patches.append(torch.tensor([p_small]))
            current_prediction.append(torch.tensor(c_p_on_patch_small))
            target_refine_net.append(torch.tensor(c_t_on_patch_small))
            target_general.append(torch.tensor(c_t_on_image))
            offsets.append([[x_start, y_start], [np.zeros(p.shape), np.zeros(p_small.shape)]])

            # fig, ax = plt.subplots(1, 1)
            # ax.imshow(p_small, cmap="binary_r")
            # ax.scatter(c_t_on_patch_small[0].detach().to("cpu").numpy(), c_t_on_patch_small[1].detach().to("cpu").numpy(), marker="^", c="green")
            # ax.scatter(c_p_on_patch_small[0].detach().to("cpu").numpy(), c_p_on_patch_small[1].detach().to("cpu").numpy(), marker="o", c="red")
            # fig.savefig(f"init_crop_temp{temp_count:02d}.png")
            # temp_count += 1
            # plt.close()

        patches = torch.stack(patches).to(self.device)
        current_prediction = torch.stack(current_prediction).to(self.device)
        target_refine_net = torch.stack(target_refine_net).to(self.device)
        target_general = torch.stack(target_general).to(self.device)

        return patches, current_prediction, target_refine_net, target_general, offsets

    def evaluate(self, method, loader: DataLoader, device: str,
                 naming: str, path: str, num=2):
        joint_corner_net = self.set_JCD()
        self.net.eval()
        allimages, fourcorners, singlepredictions = [], [], []
        count = 0
        for i, b in enumerate(loader):
            images, corners_targets, _, originals, corners_big = b
            images = images.to(self.device)
            predictions, _ = joint_corner_net(images)
            predictions = torch.reshape(predictions, shape=(len(predictions), 4, 2))
            corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))
            for j in range(len(images)):
                if count == num: break
                image = images[j]
                corners_pred = predictions[j]
                original = originals[j]
                big_corners_pred = torch.tensor([self.rescale_corners(c, image, original) for c in corners_pred])
                corner_big = corners_big[j]

                patches, old_prediction, target_refine_net, \
                target_general, offsets = self.initial_crop(originals=torch.stack([original]),
                                                            big_corners_pred=torch.stack([big_corners_pred]),
                                                            corners_big=torch.stack([corner_big]),
                                                            output_shape=(50, 50))

                p, o_p, t_r_n, target_general, offsets = patches[0], old_prediction[0], target_refine_net[0], target_general[0], offsets[0]
                n_p = self.net(p)
                changes = {}

                changes[f"{0:02d} offset"] = torch.tensor(offsets[0])
                changes[f"{1:02d} rescale"] = {"to": offsets[1][1], "from": offsets[1][0]}
                changes[f"{2:02d} prediction"] = torch.clone(n_p.detach())

                try:
                    n_p, changes = self.refine_loop(changes, n_p, o_p, t_r_n, p)
                except:
                    print("Loop failed")
                    pass
                sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
                sorted_keys = np.array([el[0] for el in sorted_changes])
                intermediate_predictions = {k:v for k, v in changes.items() if "prediction" in k}
                prediction_chain = []
                for k, pred_point in intermediate_predictions.items():
                    # perform all transformations before this prediction
                    position = np.argwhere(sorted_keys == k)[0][0]
                    pred_point = self.untracking_changes(sorted_changes[position:], pred_point)
                    prediction_chain.append(pred_point)

                allimages.append(original)
                fourcorners.append((corner_big, big_corners_pred))
                singlepredictions.append(prediction_chain)


                count += 1
                if count == num: break
            if count == num: break

        method(images=allimages, fourcorners=fourcorners, singlepredictions=singlepredictions,
               name_convention=naming, path=path)

    def depict_refinement(self, images, fourcorners, singlepredictions, name_convention: str, path: str) -> None:
        """
        This function saves plots and pictures of network performance
        """
        for i, im, four, pred in zip(range(len(images)), images, fourcorners, singlepredictions):
            pred = [el.detach().to("cpu").numpy() for el in pred]
            im = im[0].reshape(im.shape[-2], im.shape[-1])
            corners_trg, corners_pred = four

            fig, ax = plt.subplots(1, 2, figsize=(20 * 2, 20 + 3))
            ax[0].imshow(im, cmap="binary_r")

            colors = ['red', 'green', 'blue', 'yellow']

            # bring to tuples

            corners_pred[(corners_pred < 0).argwhere()] = 0
            corners_pred[(corners_pred > 2000).argwhere()] = 2000

            for c, col in zip(corners_trg, colors): ax[0].scatter(c[0], c[1], c=col, s=500, marker='o')
            for j, col in zip(range(4), colors):
                c = corners_pred[j]
                prev_c = corners_pred[j - 1]
                ax[0].scatter(c[0], c[1], c=col, s=500, marker='X')
                ax[0].plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")
            ax[0].set_title("True Image, Predictions - ☓, True - ◯", fontsize=40)
            pos = ax[0].get_position()
            ax[0].set_position([pos.x0, pos.y0 + pos.height * 0.1,
                                pos.width, pos.height * 0.9])
            ax[0].axis("off")


            active_corner_id = np.argwhere(np.array(colors)==self.net.net_type)[0][0]
            ax[1].imshow(im, cmap="binary_r")
            ax[1].scatter(corners_trg[active_corner_id][0],
                          corners_trg[active_corner_id][1],
                          c=colors[active_corner_id],
                          s=500, marker="o", label="Target corner")
            ax[1].scatter(corners_pred[active_corner_id][0],
                          corners_pred[active_corner_id][1],
                          c=colors[active_corner_id],
                          s=500, marker="X", label="JCD prediction")

            for sh, shot in enumerate(pred):
                alpha = np.linspace(1, .2, len(pred))[sh]
                if sh == len(shot)-1:
                    ax[1].scatter(shot[0],
                              shot[1],
                              c="cornflowerblue",
                              s=500, marker="X", label="CRN prediction")
                else:
                    ax[1].scatter(shot[0],
                                  shot[1],
                                  c="hotpink", alpha=alpha,
                                  s=200, marker=".")
            ax[1].set_title("Single corner prediction", fontsize=40)
            ax[1].axis("off")
            ax[1].legend(fontsize=30)


            fig.savefig(os.path.join(path, name_convention + "_imagenum_" + str(i).zfill(2) + ".png"), dpi=50)

            plt.close()
