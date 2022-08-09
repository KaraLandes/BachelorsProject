import operator
from pathlib import Path
import os
from Repo.detection.corners_nn.network import CornerDetector
from Repo.detection.corners_nn.train_evaluate import TrainCorner
from Repo.detection.corners_nn.dataset import CornerRealBillSet
from Repo.detection.refine_nn.train_evaluate import TrainRefine
from Repo.detection.refine_nn.dataset import RefineRealBillSet
from Repo.detection.refine_nn.network import RefineNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import numpy as np

# HYPERPARAMETERS, WHICH CODE TO RUN
DETECT_CORNERS = 1
DETECT_REFINE =0
repo = Path(os.getcwd())

########################################################################################################################
# Stage 1
# Detecting bills on real images of bills
########################################################################################################################
if DETECT_CORNERS:
    net = CornerDetector(compute_attention=True).to('cuda')
    net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", 'models',
                                                "run_80_1", 'corners_nn_on_ep17_new_best_model_32.0.pt')))

    bills = CornerRealBillSet(image_dir=os.path.join(repo, "processed_data", "realbills", "unseen"), output_shape=(80, 80), coefficient=1)
    bills_big = CornerRealBillSet(image_dir=os.path.join(repo, "processed_data", "realbills", "unseen"), output_shape=(1000, 1000), coefficient=1)
    train_class = TrainCorner("", "", "", network=net)

    loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)
    loader_big = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)

    loss = torch.nn.MSELoss()
    loss_values = []
    num = 0  # 0 or 1000
    distances_small = [[], [], [], []]
    distances_big = [[], [], [], []]
    for i, b in tqdm(enumerate(loader)):
        im, corner_trg, mask_trg = b
        im_big, trg_big = bills_big.__getitem__(i)
        corner_trg_big, mask_trg_big = trg_big

        im = im.to('cuda')
        corner_trg = corner_trg.to('cuda')
        pred, _ = net(im)
        l_box = loss(pred, corner_trg)
        total_l = l_box

        loss_values.append(total_l.detach().item())

        image = im_big
        prediction = pred[0].to("cpu").detach().numpy()
        prediction = [el if el > 0 else 0 for el in prediction]
        prediction = np.array([el if el < 63 else 63 for el in prediction])
        prediction_big = prediction*im_big.shape[-1]/im.shape[-1]
        prediction_big = np.array([[prediction_big[0], prediction_big[1]], [prediction_big[2], prediction_big[3]],
                                   [prediction_big[4], prediction_big[5]], [prediction_big[6], prediction_big[7]]])
        prediction = np.array([[prediction[0], prediction[1]], [prediction[2], prediction[3]],
                               [prediction[4], prediction[5]], [prediction[6], prediction[7]]])
        corner_trg = corner_trg[0].to("cpu").detach().numpy()
        corner_trg = np.array([[corner_trg[0], corner_trg[1]], [corner_trg[2], corner_trg[3]],
                               [corner_trg[4], corner_trg[5]], [corner_trg[6], corner_trg[7]]])

        fig, ax = plt.subplots(1, 1, figsize=(24, 24))
        ax.imshow(image, cmap="binary_r")
        colors = ['red', 'green', 'blue', 'yellow']
        for c, col in zip(corner_trg_big, colors): ax.scatter(c[0], c[1], c=col, s=1000, marker='o')
        for j, col in zip(range(4), colors):
            c = prediction_big[j]
            prev_c = prediction_big[j - 1]
            ax.scatter(c[0], c[1], c=col, s=1000, marker='X')
            ax.plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")
            distances_small[j].append(np.sqrt(sum((prediction[j]-corner_trg[j])**2)))
            distances_big[j].append(np.sqrt(sum((prediction_big[j] - corner_trg_big[j]) ** 2)))
        ax.set_title("True Image, Predictions - ☓, True - ◯", fontsize=50)
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + pos.height * 0.1,
                         pos.width, pos.height * 0.9])
        ax.axis("off")
        plt.tight_layout(pad=2)

        # fig.savefig(os.path.join("real_bills_results", 'detection', "corners", f"{num}.png"), dpi=50)
        num += 1
        plt.close('all')
        plt.cla()
        plt.clf()

    print("Max distances", im.shape[-2:])
    for i, col in enumerate(colors):
        print(col, "\t", max(distances_small[i]))

    print()
    print("Max distances", im_big.shape[-2:])
    for i, col in enumerate(colors):
        print(col, "\t", max(distances_big[i]))

    f = plt.figure()
    plt.boxplot(distances_big)
    f.savefig(os.path.join("real_bills_results", 'detection', "corners", f"big_dist_80_e17_unseen.png"), dpi=50)

if DETECT_REFINE:
    jcd = CornerDetector(compute_attention=True).to('cuda')
    jcd.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", 'models',
                                                "run_64_1", 'corners_nn_on_ep22_new_best_model_19.0.pt')))


    crn = RefineNet(net_type="red")#dummy instance
    bills = RefineRealBillSet(image_dir=os.path.join(repo, "processed_data", "realbills", "unseen"), output_shape=(64, 64),
                              coefficient=1)
    num = 1000 # 0 or 1000
    train_class = TrainRefine("", "", "", network=crn)

    loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn,
                        shuffle=False)
    best_nets = [os.path.join(repo, "progress_tracking", "detection/refine_nn", "red", 'models',
                              'corners_nn_on_ep10_new_best_model_2694.0.pt'),
                 os.path.join(repo, "progress_tracking", "detection/refine_nn", "green", 'models',
                              'corners_nn_on_ep19_new_best_model_3661.0.pt'),
                 os.path.join(repo, "progress_tracking", "detection/refine_nn", "blue", 'models',
                              'corners_nn_on_ep2_new_best_model_3794.0.pt'),
                 os.path.join(repo, "progress_tracking", "detection/refine_nn", "yellow", 'models',
                              'corners_nn_on_ep11_new_best_model_3539.0.pt')
                 ]
    colors = ["red", 'green', 'blue', 'yellow']
    distances_big =[[], [], [], []]
    for im, _, _, originals, corners_big in tqdm(loader):
        coords_pred, _ = jcd(im.to("cuda"))

        # recalculation of predicted corners
        coords_pred = torch.reshape(coords_pred, shape=(len(coords_pred), 4, 2))
        corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))

        big_corners_pred = torch.tensor([[train_class.rescale_corners(c, im, originals)
                                          for c in el] for el in coords_pred])

        refine_corners_pred = []
        for type, state_dict in zip(colors, best_nets):
            crn = RefineNet(net_type=type)
            state_dict = torch.load(state_dict)
            crn.load_state_dict(state_dict)
            train_class.net = crn

            patches, old_prediction, target_refine_net, \
            target_general, offsets = train_class.initial_crop(originals=originals,
                                                               big_corners_pred=big_corners_pred,
                                                               corners_big=corners_big,
                                                               output_shape=(16, 16),
                                                               window=200)

            p = patches[0].to(train_class.device)
            n_p = train_class.net(p)
            o_p = old_prediction[0]
            t_r_n = target_refine_net[0]
            changes = {}

            changes[f"{0:02d} offset"] = torch.tensor(offsets[0][0])
            changes[f"{1:02d} rescale"] = {"to": offsets[0][1][1], "from": offsets[0][1][0]}
            changes[f"{2:02d} prediction"] = torch.clone(n_p.detach())

            n_p, changes = train_class.refine_loop(False, None, torch.nn.MSELoss(reduction="mean"), changes, n_p, o_p, t_r_n, p)

            # when no more cropping happens I retrack the most current prediction to the original size and calculate loss
            sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
            pred_point = torch.clone(n_p.detach())
            pred_point = train_class.untracking_changes(sorted_changes, pred_point)

            refine_corners_pred.append(pred_point)

        fig, ax = plt.subplots(1, 2, figsize=(24*2+5, 24))
        ax[0].imshow(originals[0][0], cmap="binary_r")
        ax[1].imshow(originals[0][0], cmap="binary_r")

        for c, col in zip(corners_big[0], colors): ax[0].scatter(c[0], c[1], c=col, s=1000, marker='o')
        for c, col in zip(corners_big[0], colors): ax[1].scatter(c[0], c[1], c=col, s=1000, marker='o')

        for j, col in zip(range(4), colors):
            c = refine_corners_pred[j]
            prev_c = refine_corners_pred[j - 1]
            ax[0].scatter(c[0], c[1], c=col, s=1000, marker='X')
            ax[0].plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")

            distances_big[j].append(np.sqrt(sum((c - corners_big[0][j]) ** 2)))

        for j, col in zip(range(4), colors):
            c = big_corners_pred[0][j]
            prev_c = big_corners_pred[0][j - 1]
            ax[1].scatter(c[0], c[1], c=col, s=1000, marker='X')
            ax[1].plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")

        ax[0].set_title("Corners NN, Predictions - ☓, True - ◯", fontsize=50)
        ax[1].set_title("Refine NN, Predictions - ☓, True - ◯", fontsize=50)
        pos = ax[0].get_position()
        ax[0].set_position([pos.x0, pos.y0 + pos.height * 0.1,
                         pos.width, pos.height * 0.9])
        ax[0].axis("off")
        pos = ax[1].get_position()
        ax[1].set_position([pos.x0, pos.y0 + pos.height * 0.1,
                            pos.width, pos.height * 0.9])
        ax[1].axis("off")
        plt.tight_layout(pad=3)

        fig.savefig(os.path.join("real_bills_results", 'detection', "refine", f"{num}.png"), dpi=50)

        plt.close('all')
        plt.cla()
        plt.clf()

        num += 1

    print()
    print("Max distances", originals.shape[-2:])
    for i, col in enumerate(colors):
        print(col, "\t", max(distances_big[i]))

    f = plt.figure()
    plt.boxplot(distances_big)
    f.savefig(os.path.join("real_bills_results", 'detection', "refine", f"big_dist_unseen.png"), dpi=50)
