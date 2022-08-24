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
DETECT_CORNERS = 0
DETECT_REFINE = 1
repo = Path(os.getcwd())

########################################################################################################################
# Stage 1
# Detecting bills on real images of bills
########################################################################################################################
if DETECT_CORNERS:
    net = CornerDetector(compute_attention=True).to('cuda')
    net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", 'models',
                                                "run_80_crop_1", "retrain_1",
                                                'corners_nn_on_ep7_new_best_model_26.0.pt')))

    bills = CornerRealBillSet(image_dir=os.path.join(repo, "processed_data", "faster_processed_2", "realbills",
                                                     "unseen"),
                              output_shape=(80, 80), coefficient=1)
    bills_big = CornerRealBillSet(image_dir=os.path.join(repo, "processed_data", "faster_processed_2", "realbills",
                                                         "unseen"),
                                  output_shape=(1000, 1000), coefficient=1)
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
        temp = np.zeros((image.shape[1], image.shape[1], 3))
        temp[:, :, 0] = image[0]
        temp[:, :, 1] = image[1]
        temp[:, :, 2] = image[2]
        ax.imshow(temp.astype(int))
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

        fig.savefig(os.path.join("real_bills_results", 'detection', "corners", f"{num}.png"), dpi=50)
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
    plt.ylim((0, 300))
    f.savefig(os.path.join("real_bills_results", 'detection', "corners", f"big_dist_80_crop_retrain_unseen.png"),
              dpi=100)

if DETECT_REFINE:
    jcd = CornerDetector(compute_attention=True).to('cuda')
    jcd.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", 'models',
                                                "run_48_crop", "retrain_2",
                                                'corners_nn_on_ep27_new_best_model_10.0.pt')))


    crn = RefineNet(net_type="red")#dummy instance
    bills = RefineRealBillSet(image_dir=os.path.join(repo, "processed_data", "faster_processed_2",
                                                     "realbills", "unseen"), output_shape=(48, 48),
                              coefficient=1)
    num = 1000 # 0 or 1000
    train_class = TrainRefine("", "", "", network=crn)
    train_class.set_device()
    loader = DataLoader(dataset=bills, batch_size=1, num_workers=11, collate_fn=train_class.collate_fn,
                        shuffle=False)
    best_nets = [os.path.join(repo, "progress_tracking", "detection/refine_nn", "red", 'models',
                              'corners_nn_on_ep19_new_best_model_68.0.pt'),
                 os.path.join(repo, "progress_tracking", "detection/refine_nn", "green", 'models',
                              'corners_nn_on_ep20_new_best_model_73.0.pt'),
                 os.path.join(repo, "progress_tracking", "detection/refine_nn", "blue", 'models',
                              'corners_nn_on_ep20_new_best_model_83.0.pt'),
                 os.path.join(repo, "progress_tracking", "detection/refine_nn", "yellow", 'models',
                              'corners_nn_on_ep20_new_best_model_64.0.pt')
                 ]
    colors = ["red", 'green', 'blue', 'yellow']
    distances_big =[[], [], [], []]
    for im, _, _, originals, corners_big in tqdm(loader):
        coords_pred, _ = jcd(im.to("cuda"))

        # recalculation of predicted corners
        coords_pred = torch.reshape(coords_pred, shape=(len(coords_pred), 4, 2))
        corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))

        big_corners_pred = torch.stack([torch.stack([train_class.rescale_corner(c, im.shape[-2:], originals.shape[-2:])
                                                     for c in el]) for el in coords_pred])

        refine_corners_pred = []
        for active_id, type, state_dict in zip(range(len(colors)), colors, best_nets):
            crn = RefineNet(net_type=type)
            state_dict = torch.load(state_dict)
            crn.load_state_dict(state_dict)
            train_class.net = crn.to(train_class.device)

            p, _, changes = train_class.crop(original=train_class.numpy(originals[0]),
                                             old_prediction=train_class.numpy(big_corners_pred[0][active_id]),
                                             changes={},
                                             mask=np.zeros(originals[0].shape[-2:]))
            sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=False)
            p = torch.tensor(p).to(train_class.device)

            o_p_big = big_corners_pred[0][active_id]
            o_p_for_net = train_class.apply_changes(o_p_big, sorted_changes)

            o_p_for_net = o_p_for_net / p.shape[-1]  # relative
            n_p, mask_pred = train_class.net((p, o_p_for_net))
            n_p_on_patch = n_p * p.shape[-2]
            sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
            n_p_big = train_class.untrack_changes(n_p_on_patch, sorted_changes)

            pred_point, _ = train_class.refine_loop(n_p_big=n_p_big,
                                                    o_p_big=o_p_big,
                                                    target=corners_big[0][active_id],
                                                    changes=changes,
                                                    patch=p)

            #TODO al this prediction cycle for every net type
            refine_corners_pred.append(train_class.numpy(pred_point))

        # predictions averaging
        beta = [1, 1, 1, 1]
        averaged_prediction = []
        for rc, bc, bt in zip(refine_corners_pred, train_class.numpy(big_corners_pred[0]), beta):
            c = rc*bt + bc*(1-bt)
            averaged_prediction.append(c)
        averaged_prediction = np.array(averaged_prediction)

        fig, ax = plt.subplots(1, 2, figsize=(24*2+5, 24))
        temp = np.zeros((originals[0].shape[1], originals[0].shape[1], 3))
        temp[:, :, 0] = originals[0][0]
        temp[:, :, 1] = originals[0][1]
        temp[:, :, 2] = originals[0][2]
        ax[0].imshow(temp)
        ax[1].imshow(temp)

        for c, col in zip(corners_big[0], colors): ax[0].scatter(c[0], c[1], c=col, s=1000, marker='o')
        for c, col in zip(corners_big[0], colors): ax[1].scatter(c[0], c[1], c=col, s=1000, marker='o')

        for j, col in zip(range(4), colors):
            c = averaged_prediction[j]
            prev_c = averaged_prediction[j - 1]
            ax[1].scatter(c[0], c[1], c=col, s=1000, marker='X')
            ax[1].plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")

            distances_big[j].append(np.sqrt(sum((c - train_class.numpy(corners_big[0][j])) ** 2)))

        big_corners_pred = train_class.numpy(big_corners_pred)
        for j, col in zip(range(4), colors):
            c = big_corners_pred[0][j]
            prev_c = big_corners_pred[0][j - 1]
            ax[0].scatter(c[0], c[1], c=col, s=1000, marker='X')
            ax[0].plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")

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
    plt.ylim((0, 300))
    f.savefig(os.path.join("real_bills_results", 'detection', "refine", f"big_dist_unseen_beta_{beta}.png"), dpi=50)
