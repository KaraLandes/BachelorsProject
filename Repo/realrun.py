from pathlib import Path
import os
from Repo.detection.corners_nn.network import CornerDetector
from Repo.detection.corners_nn.train_evaluate import TrainCorner
from Repo.detection.corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
import numpy as np

# HYPERPARAMETERS, WHICH CODE TO RUN
DETECT = True
repo = Path(os.getcwd())

########################################################################################################################
# Stage 1
# Detecting bills on real images of bills
########################################################################################################################
if DETECT:
    net = CornerDetector(compute_attention=True).to('cuda')
    net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", 'models',
                                                "run_256_2", 'corners_nn_on_ep15_new_best_model_370.0.pt')))

    bills = CornerRealBillSet(image_dir=os.path.join(repo, "processed_data", "realbills", "unseen"), output_shape=(256, 256), coefficient=1)
    train_class = TrainCorner("", "", "", network=net)
    loader = DataLoader(dataset=bills, batch_size=1, num_workers=2, collate_fn=train_class.collate_fn)

    loss = torch.nn.MSELoss()
    loss_values = []
    num = 1000  # 0 or 1000
    distances = [[], [], [], []]
    for im, corner_trg, mask_trg in tqdm(loader):
        im = im.to('cuda')
        corner_trg = corner_trg.to('cuda')
        pred, _ = net(im)
        l_box = loss(pred, corner_trg)
        total_l = l_box

        loss_values.append(total_l.detach().item())

        im = im.to('cpu').detach().numpy()
        corner_trg = corner_trg.to('cpu').detach().numpy()
        pred = pred.to('cpu').detach().numpy()
        for i, image, prediction, target in zip(range(len(im)), im, pred, corner_trg):
            image = image[0]
            prediction = [el if el > 0 else 0 for el in prediction]
            prediction = [el if el < 127 else 127 for el in prediction]
            prediction = np.array([[prediction[0], prediction[1]], [prediction[2], prediction[3]],
                                  [prediction[4], prediction[5]], [prediction[6], prediction[7]]])
            target = np.array([[target[0], target[1]], [target[2], target[3]],
                              [target[4], target[5]], [target[6], target[7]]])

            fig, ax = plt.subplots(1, 1, figsize=(24, 24))
            ax.imshow(image, cmap="binary_r")
            colors = ['red', 'green', 'blue', 'yellow']
            for c, col in zip(target, colors): ax.scatter(c[0], c[1], c=col, s=1000, marker='o')
            for j, col in zip(range(4), colors):
                c = prediction[j]
                prev_c = prediction[j - 1]
                ax.scatter(c[0], c[1], c=col, s=1000, marker='X')
                ax.plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")
                distances[j].append(np.sqrt(sum((prediction[j]-target[j])**2)))
            ax.set_title("True Image, Predictions - ☓, True - ◯", fontsize=50)
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 + pos.height * 0.1,
                             pos.width, pos.height * 0.9])
            ax.axis("off")
            plt.tight_layout(pad=2)

            fig.savefig(os.path.join("real_bills_results", 'detection', "corners", f"{num}.png"), dpi=200)
            num += 1
            plt.close('all')
            plt.cla()
            plt.clf()

    print("Max distances")
    for i, col in enumerate(colors):
        print(col, "\t", max(distances[i]))

