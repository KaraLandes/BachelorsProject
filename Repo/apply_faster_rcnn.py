from pathlib import Path
import os

import pandas as pd

from Repo.detection.faster_rcnn.train_evaluate import TrainFRCNN
from Repo.detection.faster_rcnn.dataset import FRCNNRealBillSet
from Repo.detection.dummy_cnn.dataset import BaseRealBillSet
from Repo.detection.faster_rcnn.network import FasterRCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

repo = Path(os.getcwd()).parent

im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")

im_dir_real_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills")
im_dir_unseen_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

save_dir = os.path.join(repo, "processed_data", "faster_processed", "realbills")
save_dir_unseen = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

save_dir_faster = os.path.join(repo, "processed_data", "faster_processed_2", "realbills")
save_dir_unseen_faster = os.path.join(repo, "processed_data", "faster_processed_2", "realbills", "unseen")

UNSEEN = 0  # test or train data
ORIGINAL = 1  # original or cropped previously data
SHAPE = 64

if ORIGINAL:
    if UNSEEN:
        im_dir = im_dir_unseen
        save_dir = save_dir_unseen
    else:
        im_dir = im_dir_real
        save_dir = save_dir
else:
    if UNSEEN:
        im_dir = im_dir_unseen_faster
        save_dir = save_dir_unseen_faster
    else:
        im_dir = im_dir_real_faster
        save_dir = save_dir_faster

net = FasterRCNN()
net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/faster_rcnn", 'models',
                                            str(SHAPE),
                                            "faster_rcnn__on_ep8_new_best_model_25.0.pt")))

bills = FRCNNRealBillSet(image_dir=im_dir, output_shape=(SHAPE, SHAPE), coefficient=1)
bills_big = BaseRealBillSet(image_dir=im_dir, output_shape=(3000, 3000), coefficient=1)

train_class = TrainFRCNN("", "", "", network=net)
train_class.set_device()
net = net.to(train_class.device)

loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)
loader_big = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)

errors = {}
errors["xmin0"] = []  # 0 absolute error
errors["xmin1"] = []  # 1 normalised error
errors["ymin0"] = []
errors["ymin1"] = []
errors["xmax0"] = []
errors["xmax1"] = []
errors["ymax0"] = []
errors["ymax1"] = []
areas = []
for i, b in tqdm(enumerate(loader)):
    im, trg = b
    im_big, trg_big = bills_big.__getitem__(i)

    im = [el.to(train_class.device) for el in im]
    trg = [{k: v.to(train_class.device) for k, v in t.items()} for t in trg]

    pred, _ = train_class.net((im, trg, False))
    pred = pred[0]['boxes'][0].detach().cpu().numpy()
    pred_big = ((pred / im[0].shape[1]) * im_big.shape[1]).astype(int)

    pred_big[1], pred_big[2] = pred_big[2], pred_big[1]

    pred_big_w = abs(pred_big[0] - pred_big[1])
    pred_big_h = abs(pred_big[2] - pred_big[3])
    r_tol_w, r_tol_h = 0.25, 0.25  #
    w_tolerance = int(pred_big_w * r_tol_w)
    h_tolerance = int(pred_big_h * r_tol_h)
    pred_big[0] = 0 if pred_big[0] - w_tolerance < 0 else pred_big[0] - w_tolerance
    pred_big[2] = 0 if pred_big[2] - h_tolerance < 0 else pred_big[2] - h_tolerance
    pred_big[1] = im_big.shape[-2] if pred_big[1] + w_tolerance > im_big.shape[-2] else pred_big[1] + w_tolerance
    pred_big[3] = im_big.shape[-2] if pred_big[3] + h_tolerance > im_big.shape[-2] else pred_big[3] + h_tolerance

    # now i need to ensure cropped image has proportions as I need
    pred_big_w = abs(pred_big[0] - pred_big[1])
    pred_big_h = abs(pred_big[2] - pred_big[3])
    proportion = 4032 / 1816
    adjust = 0  # if adjust = 0 then I need to do smth width, 1-> with height
    if pred_big_w >= pred_big_h:
        desired_h = pred_big_w / proportion
        delta = desired_h - pred_big_h
        if delta < 0:  adjust = 0
        else: adjust = 1
    else:
        desired_w = pred_big_h / proportion
        delta = desired_w - pred_big_w
        if delta < 0: adjust = 1
        else: adjust = 0

    if adjust==0:
        desired_w = proportion*pred_big_h if pred_big_h<=pred_big_w else pred_big_h/proportion
        delta = np.ceil(desired_w-pred_big_w).astype(int)
        pixels_to_left, pixels_to_right = pred_big[0], 3000-pred_big[1]
        half_delta = np.ceil(delta/2).astype(int)
        if half_delta <= pixels_to_left and half_delta <= pixels_to_right:
            pred_big[0] = pred_big[0]-half_delta
            pred_big[1] = pred_big[1] + half_delta
        elif half_delta > pixels_to_left and half_delta <= pixels_to_right:
            pred_big[0] = 0
            delta = int(delta-pixels_to_left)
            if delta < pixels_to_right: pred_big[1] = pred_big[1] + delta
            else: pred_big[1] = 3000
        elif half_delta <= pixels_to_left and half_delta > pixels_to_right:
            pred_big[1] = 3000
            delta = int(delta-pixels_to_right)
            if delta < pixels_to_left: pred_big[0] = pred_big[0] - delta
            else: pred_big[0] = 0
    else:
        desired_h = proportion*pred_big_w if pred_big_w<=pred_big_h else pred_big_w/proportion
        delta = np.ceil(desired_h-pred_big_h).astype(int)
        pixels_to_top, pixels_to_bottom = pred_big[2], 3000-pred_big[3]
        half_delta = np.ceil(delta/2).astype(int)
        if half_delta <= pixels_to_top and half_delta <= pixels_to_bottom:
            pred_big[2] = pred_big[2]-half_delta
            pred_big[3] = pred_big[3] + half_delta
        elif half_delta > pixels_to_top and half_delta <= pixels_to_bottom:
            pred_big[2] = 0
            delta = int(delta-pixels_to_top)
            if delta < pixels_to_bottom: pred_big[3] = pred_big[3] + delta
            else: pred_big[3] = 3000
        elif half_delta <= pixels_to_top and half_delta > pixels_to_bottom:
            pred_big[3] = 3000
            delta = int(delta-pixels_to_bottom)
            if delta < pixels_to_top: pred_big[2] = pred_big[2] - delta
            else: pred_big[2] = 0

    cropped_image = im_big[pred_big[0]:pred_big[1], pred_big[2]:pred_big[3], :]
    cropped_mask = trg_big["masks"][pred_big[0]:pred_big[1], pred_big[2]:pred_big[3]]

    cropped_image = cropped_image.astype(np.uint8)
    Image.fromarray(cropped_image).save(os.path.join(save_dir, f"real_image_{i:03d}.jpg"))
    Image.fromarray(cropped_mask.astype(np.uint8)).save(os.path.join(save_dir, f"mask_real_image_{i:03d}.jpg"))

    # cropped area tracking
    pred_big_w = abs(pred_big[0] - pred_big[1])
    pred_big_h = abs(pred_big[2] - pred_big[3])
    discarded_area = (3000**2)-(pred_big_h*pred_big_w)
    areas.append(discarded_area)
    # errors tracking
    # on image 3000 x 3000 absolute and relative error
    x_min_diff = abs(pred_big[0] - trg_big['boxes'][0])
    y_min_diff = abs(pred_big[1] - trg_big['boxes'][2])
    x_max_diff = abs(pred_big[2] - trg_big['boxes'][1])
    y_max_diff = abs(pred_big[3] - trg_big['boxes'][3])

    errors["xmin0"].append(x_min_diff)
    errors["xmin1"].append(x_min_diff / 3000)
    errors["xmax0"].append(x_max_diff)
    errors["xmax1"].append(x_max_diff / 3000)
    errors["ymin0"].append(y_min_diff)
    errors["ymin1"].append(y_min_diff / 3000)
    errors["ymax0"].append(y_max_diff)
    errors["ymax1"].append(y_max_diff / 3000)

    # check mask
    # temp_i = np.zeros((im_big.shape[-2], im_big.shape[-1], 3))
    # temp_i[:, :, 0] = im_big[0] * trg_big["masks"].detach().cpu().numpy()[0]
    # temp_i[:, :, 1] = im_big[1] * trg_big["masks"].detach().cpu().numpy()[0]
    # temp_i[:, :, 2] = im_big[2] * trg_big["masks"].detach().cpu().numpy()[0]
    # temp_i = temp_i.astype(np.uint8)
    # Image.fromarray(temp_i).save(os.path.join(save_dir, f"checkup_{i}.jpg"))

# save errors
frame = pd.DataFrame(errors)
frame.to_csv(os.path.join(save_dir, f"frcnn_errors_{'test' if UNSEEN else 'train'}_{SHAPE}.csv"), sep=";")

print("Average areas discarded")
print(np.mean(areas))
print(np.mean(areas)/(3000**2))