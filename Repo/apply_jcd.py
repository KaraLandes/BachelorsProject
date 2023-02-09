from pathlib import Path
import os

import pandas as pd
from matplotlib import pyplot as plt

from Repo.detection.corners_nn.network import CornerDetector
from Repo.detection.corners_nn.train_evaluate import TrainCorner
from Repo.detection.corners_nn.dataset import CornerRealBillSet
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

repo = os.path.join(Path(os.getcwd()).parent, "Repo")

im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")

im_dir_real_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills")
im_dir_unseen_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

save_dir = os.path.join(repo, "progress_tracking", "detection/corners_nn_jcd", "errors")
save_dir_c = os.path.join(repo, "progress_tracking", "detection/corners_nn_c_jcd", "errors")

UNSEEN = 0 # test or train data
ORIGINAL = 0 # original or cropped previously data
SHAPE = 256
HIGHRES = 1000

if ORIGINAL:
    if UNSEEN:
        im_dir = im_dir_unseen
    else:
        im_dir = im_dir_real
else:
    if UNSEEN:
        im_dir = im_dir_unseen_faster
    else:
        im_dir = im_dir_real_faster
    save_dir = save_dir_c

net = CornerDetector(compute_attention=True, size=SHAPE).to('cpu')
net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn_c_jcd", 'models',
                                            str(SHAPE),
                                            "corners_nn_on_ep10_new_best_model_159.0.pt")))

bills = CornerRealBillSet(image_dir=im_dir, output_shape=(SHAPE, SHAPE), coefficient=1)
bills_big = CornerRealBillSet(image_dir=im_dir, output_shape=(HIGHRES, HIGHRES), coefficient=1)

train_class = TrainCorner("", "", "", network=net)
train_class.set_device(cpu=True)
net = net.to(train_class.device)

loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)
loader_big = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)

errors = {}
errors["c1_dist"] = []
errors["c1_dist_n"] = []
errors["c2_dist"] = []
errors["c2_dist_n"] = []
errors["c3_dist"] = []
errors["c3_dist_n"] = []
errors["c4_dist"] = []
errors["c4_dist_n"] = []

for i, b in tqdm(enumerate(loader)):
    im, corner_trg, mask_trg = b
    im_big, trg_big = bills_big.__getitem__(i)
    corner_trg_big, mask_trg_big = trg_big

    im = im.to('cpu')
    corner_trg = corner_trg.to('cpu')
    pred, _ = net(im)
    prediction = pred[0].to("cpu").detach().numpy()
    prediction = [el if el > 0 else 0 for el in prediction]
    prediction = np.array([el if el < SHAPE-1 else SHAPE-1 for el in prediction])
    pred_big = prediction * im_big.shape[-1] / im.shape[-1]
    pred_big = pred_big.reshape(4, 2)


    # errors tracking
    # on image 1000x1000 absolute and relative eucledian distances
    c1 = np.linalg.norm(pred_big[0] - trg_big[0][0])
    c2 = np.linalg.norm(pred_big[1] - trg_big[0][1])
    c3 = np.linalg.norm(pred_big[2] - trg_big[0][2])
    c4 = np.linalg.norm(pred_big[3] - trg_big[0][3])

    errors["c1_dist"].append(c1)
    errors["c1_dist_n"].append(c1 / HIGHRES)
    errors["c2_dist"].append(c2)
    errors["c2_dist_n"].append(c2 / HIGHRES)
    errors["c3_dist"].append(c3)
    errors["c3_dist_n"].append(c3 / HIGHRES)
    errors["c4_dist"].append(c4)
    errors["c4_dist_n"].append(c4 / HIGHRES)

    # plot and save results
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    temp = np.zeros((HIGHRES, HIGHRES, 3))
    temp[:, :, 0] = im_big[0]
    temp[:, :, 1] = im_big[1]
    temp[:, :, 2] = im_big[2]
    im_big = np.array(temp)/255
    ax.imshow(im_big, cmap="binary_r")
    colors = ['r', 'g', 'b', 'y']
    marker_size = 500
    for c, col in zip(corner_trg_big, colors):
        if col == "r":
            ax.scatter(c[0], c[1], c=col, s=marker_size, marker='o', label="true corners")
        else:
            ax.scatter(c[0], c[1], c=col, s=marker_size, marker='o')
    for j, col in zip(range(4), colors):
        c = pred_big[j]
        prev_c = pred_big[j - 1]
        if col == "r":
            ax.scatter(c[0], c[1], c=col, s=marker_size, marker='X', label="prediction")
        else:
            ax.scatter(c[0], c[1], c=col, s=marker_size, marker='X')
        ax.plot((c[0], prev_c[0]), (c[1], prev_c[1]), c="red", ls=":")
    ax.legend(fontsize="xx-large")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pics", f"{'test' if UNSEEN else 'train'}_image_{i}.png"), dpi=150)
    plt.close("all")


#save errors
frame = pd.DataFrame(errors)
frame.to_csv(os.path.join(save_dir, f"jcd_c_errors_{'test' if UNSEEN else 'train'}_{SHAPE}.csv"), sep=";")


