import operator
import os
from pathlib import Path

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

        # collated_originals = [el.expand(3, el.shape[-2], el.shape[-1]) for el in collated_originals]
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
        for im, _, masks_trg, originals, corners_big in tqdm(loader):
            # first we predict corners for images with previously trained net
            im = im.to(self.device)
            coords_pred, _ = joint_corner_net(im)

            # recalculation of predicted corners to original image system
            coords_pred = torch.reshape(coords_pred, shape=(len(coords_pred), 4, 2))
            corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))

            big_corners_pred = torch.stack([torch.stack([self.rescale_corner(c, im.shape[-2:], originals.shape[-2:])
                                              for c in el]) for el in coords_pred])

            loss_values_per_sample = []
            colors = np.array(['red', 'green', 'blue', 'yellow'])
            active_corner_id = np.argwhere(colors == self.net.net_type)[0][0]
            for j in range(len(im)):
                mask = self.numpy(masks_trg[j])
                # perform initial crop
                p, mask_t, changes = self.crop(original=self.numpy(originals[j]),
                                               old_prediction=self.numpy(big_corners_pred[j][active_corner_id]),
                                               changes={},
                                               mask=mask)

                sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=False)
                p = torch.tensor(p).to(self.device)
                mask_t = torch.tensor(mask_t).to(self.device)

                # translate previous prediction and target to patch system of coordinates
                o_p_big = big_corners_pred[j][active_corner_id]
                o_p_for_net = self.apply_changes(o_p_big, sorted_changes)

                # predict
                o_p_for_net = o_p_for_net/p.shape[-1]  # relative
                # radial_mask = self.radial_mask(p.shape[-2:], self.numpy(o_p_for_net))  # radial mask
                n_p, mask_p = self.net((p, o_p_for_net))

                #network predicts relative coordinate on a patch, i translate it to system
                n_p_on_patch = n_p * p.shape[-2]

                # translate new prediction to original system of coordinates
                sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
                n_p_big = self.untrack_changes(n_p_on_patch, sorted_changes)

                if optimize:
                    n_p_big, _ = self.refine_loop(n_p_big=n_p_big,
                                                  o_p_big=o_p_big,
                                                  target=corners_big[j][active_corner_id],
                                                  optimizer=optimizer,
                                                  optimize=optimize,
                                                  criterion=criterion,
                                                  criterion2=criterion2,
                                                  changes=changes,
                                                  patch=p,
                                                  mask_t=mask_t,
                                                  mask_p=mask_p)
                else:
                    n_p_big, _ = self.refine_loop(n_p_big=n_p_big,
                                                  o_p_big=o_p_big,
                                                  target=corners_big[j][active_corner_id],
                                                  changes=changes,
                                                  patch=p)

                # loss_values_per_sample.append(criterion(pred_point, target_general[0]).item())
                loss_values_per_sample.append((sum((n_p_big.cpu() - corners_big[j][active_corner_id]) ** 2)**0.5).item())

            loss_values.append(np.mean(loss_values_per_sample))
            self.writer.add_scalars("Current_epoch_progress",
                                    {
                                        "Loss_of_corners": np.mean(loss_values[-1]),
                                    },
                                    global_step=i)
            self.writer.flush()
            i += 1

        if not optimize:
            print()
            print("Stats:")
            print("min\t", min(loss_values))
            print("max\t", max(loss_values))
            print("quantiles", np.quantile(loss_values, [.25,.5,.75]))
            print()
        return loss_values

    def radial_mask(self, size, center):
        arr = np.zeros(size, dtype=np.uint8)
        imgsize = arr.shape[:2]
        innerColor = 255
        outerColor = 0
        for y in range(imgsize[0]):
            for x in range(imgsize[1]):
                # Find the distance to the center
                distanceToCenter = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                # Make it on a scale from 0 to 1innerColor
                distanceToCenter = distanceToCenter / size[0]
                # Calculate r, g, and b values
                r = outerColor * distanceToCenter + innerColor * (1 - distanceToCenter)
                arr[y, x] = r

        return arr

    def refine_loop_step(self, n_p_big, o_p_big, count, p, changes):
        alpha = 1
        o_p_big_weighted = n_p_big*alpha + o_p_big*(1-alpha)
        sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=False)
        o_p_on_patch = self.apply_changes(o_p_big_weighted, sorted_changes)

        # predict
        o_p_for_net = o_p_on_patch / p.shape[-1]  # relative
        # radial_mask = self.radial_mask(p.shape[-2:], self.numpy(o_p_on_patch))  # radial mask
        n_p, mask_p = self.net((p, o_p_for_net))

        # network predicts relative coordinate on a patch, I translate it to system
        n_p_on_patch = n_p * p.shape[-2]

        # f, a = plt.subplots(1, 2)
        # a, a1 = a
        # temp = np.zeros((self.numpy(p).shape[1], self.numpy(p).shape[1], 3))
        # temp[:, :, 0] = self.numpy(p)[0]
        # temp[:, :, 1] = self.numpy(p)[1]
        # temp[:, :, 2] = self.numpy(p)[2]
        # a.imshow(temp)
        # a.scatter(self.numpy(o_p_on_patch)[0], self.numpy(o_p_on_patch)[1], s=50, marker="X", c="y", label="OLD")
        # a.scatter(self.numpy(n_p_on_patch)[0], self.numpy(n_p_on_patch)[1], s=50, marker="o", c="r", label="NEW")
        # a.legend()
        # a1.imshow(self.numpy(mask_p[0]))
        # f.savefig(f"patch_{count}")
        # plt.close()

        # translate new prediction to original system of coordinates
        sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
        n_p_big = self.untrack_changes(n_p_on_patch, sorted_changes)

        return n_p_big, o_p_big_weighted, mask_p

    def refine_loop(self, n_p_big, o_p_big, target, changes, patch, mask_t=None, mask_p=None,
                    optimizer=None, optimize=False, criterion=None, criterion2=None):

        track = []
        track_mask = []
        count = 0
        condition1 = sum((n_p_big - o_p_big.to(self.device)) ** 2)**0.5
        condition2 = count < 1 if optimize else True

        while condition1 and condition2:
            if optimize:
                loss = criterion(n_p_big, target.to(self.device))
                loss_mask = criterion2(torch.flatten(mask_p[0]), torch.flatten(mask_t).to(torch.long))
                optimizer.zero_grad()
                (loss+loss_mask).backward()
                optimizer.step()
            n_p_big, o_p_big, mask_p = self.refine_loop_step(n_p_big=n_p_big,
                                                             o_p_big=o_p_big,
                                                             count=count,
                                                             changes=changes,
                                                             p=patch)


            track.append(n_p_big)
            count += 1
            condition1 = sum((n_p_big - o_p_big) ** 2) ** 0.5 > 1
            condition2 = count < 1 if optimize else count < 30

        return n_p_big, track

    def untrack_changes(self, point, sorted_changes):
        for k, v in sorted_changes:
            if "offset" in k:
                point = point.to(self.device) + torch.tensor(v).to(self.device)
            elif "rescale" in k:
                point = self.rescale_corner(point, from_shape=v["to"], to_shape=v["from"])
        return point

    def apply_changes(self, point, sorted_changes):
        for k, v in sorted_changes:
            if "offset" in k:
                point = torch.tensor(point).to(self.device) - torch.tensor(v).to(self.device)
            elif "rescale" in k:
                point = self.rescale_corner(point, from_shape=v["from"], to_shape=v["to"])
        return point

    def crop(self, original, old_prediction, changes, mask, window=150, p_size=(48, 48), additional_offset=(0, 0)):
        x_start = 0 if old_prediction[0] - window < 0 else old_prediction[0] - window
        x_start = x_start+additional_offset[0] if x_start+additional_offset[0]>0 else 0
        y_start = 0 if old_prediction[1] - window < 0 else old_prediction[1] - window
        y_start = y_start+additional_offset[1] if y_start+additional_offset[1]>0 else 0
        x_end = original.shape[-2] if old_prediction[0] + window > original.shape[-2] else old_prediction[0] + window
        x_end = x_end+additional_offset[0] if x_end+additional_offset[0]<original.shape[-2] else original.shape[-2]
        y_end = original.shape[-2] if old_prediction[1] + window > original.shape[-2] else old_prediction[1] + window
        y_end = y_end+additional_offset[1] if y_end+additional_offset[1]<original.shape[-2] else original.shape[-2]

        x_start, x_end, y_start, y_end = int(x_start), int(x_end), int(y_start), int(y_end)

        patch = original[:, y_start:y_end, x_start:x_end]
        patch_small = np.array([cv2.resize(ch, p_size, interpolation=cv2.INTER_AREA) for ch in patch])

        mask_patch = mask[y_start:y_end, x_start:x_end]
        mask_patch = cv2.resize(mask_patch, p_size, interpolation=cv2.INTER_AREA)

        changes[f"{0:02d} offset"] = [x_start, y_start]
        changes[f"{1:02d} rescale"] = {"to": p_size, "from": patch.shape[-2:]}

        return patch_small, mask_patch, changes

    def rescale_corner(self, corner, from_shape, to_shape):
        corner = corner/from_shape[0]
        corner = corner*to_shape[0]
        return corner

    def numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def item(self, tensor):
        return tensor.detach().cpu().item()

    def set_JCD(self):
        repo = Path(os.getcwd())
        jcd = CornerDetector(compute_attention=True).to('cuda')
        jcd.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", 'models',
                                                    "run_48_crop", "retrain_2",
                                                    'corners_nn_on_ep27_new_best_model_10.0.pt')))
        return jcd

    def evaluate(self, method, loader: DataLoader, device: str,
                 naming: str, path: str, num=2):
        joint_corner_net = self.set_JCD()
        self.net.eval()
        allimages, fourcorners, singlepredictions = [], [], []
        count = 0
        for i, b in enumerate(loader):
            im = b[0]
            _, masks_trg, originals, corners_big = b[1:]
            # first we predict corners for images with previously trained net
            im = im.to(self.device)
            coords_pred, _ = joint_corner_net(im)

            # recalculation of predicted corners to original image system
            coords_pred = torch.reshape(coords_pred, shape=(len(coords_pred), 4, 2))
            corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))

            big_corners_pred = torch.stack([torch.stack([self.rescale_corner(c, im.shape[-2:], originals.shape[-2:])
                                                         for c in el]) for el in coords_pred])

            colors = np.array(['red', 'green', 'blue', 'yellow'])
            active_corner_id = np.argwhere(colors == self.net.net_type)[0][0]
            for j in range(len(im)):
                mask = self.numpy(masks_trg[j])
                # perform initial crop
                p, _, changes = self.crop(original=self.numpy(originals[j]),
                                       old_prediction=self.numpy(big_corners_pred[j][active_corner_id]),
                                       changes={},
                                       mask=mask)

                sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=False)
                p = torch.tensor(p).to(self.device)

                # translate previous prediction and target to patch system of coordinates
                o_p_big = big_corners_pred[j][active_corner_id]
                o_p_for_net = self.apply_changes(o_p_big, sorted_changes)

                # predict
                o_p_for_net = o_p_for_net / p.shape[-1]  # relative
                # radial_mask = self.radial_mask(p.shape[-2:], self.numpy(o_p_for_net))  # radial mask
                n_p, mask_pred = self.net((p, o_p_for_net))

                # network predicts relative coordinate on a patch, I translate it to system
                n_p_on_patch = n_p * p.shape[-2]

                # translate new prediction to original system of coordinates
                sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
                n_p_big = self.untrack_changes(n_p_on_patch, sorted_changes)

                _, track = self.refine_loop(n_p_big=n_p_big,
                                            o_p_big=o_p_big,
                                            target=corners_big[j][active_corner_id],
                                            changes=changes,
                                            patch=p)
                allimages.append(originals[j])
                fourcorners.append((corners_big[j], big_corners_pred[j]))
                singlepredictions.append(track)

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

            temp = np.zeros((im.shape[1], im.shape[1], 3))
            temp[:, :, 0] = im[0]
            temp[:, :, 1] = im[1]
            temp[:, :, 2] = im[2]
            im = np.array(temp)
            corners_trg, corners_pred = four
            corners_trg, corners_pred = self.numpy(corners_trg), self.numpy(corners_pred)

            fig, ax = plt.subplots(1, 2, figsize=(20 * 2, 20 + 3))
            ax[0].imshow(im)

            colors = ['red', 'green', 'blue', 'yellow']

            # bring to tuples

            corners_pred[np.argwhere(corners_pred < 0)] = 0
            corners_pred[np.argwhere(corners_pred > im.shape[-2])] = im.shape[-2]

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
            ax[1].imshow(im)
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
